"""Build the research note and figures from the preserved run index.

Python 3.10+; dependencies: matplotlib, numpy, Pillow.
Run from any directory: python paper/build_paper.py
Produces a vector PDF with embedded fonts, figures, and matching PNG previews.
No training, network access, or model loading is performed.
"""
from pathlib import Path
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import textwrap
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw
import numpy as np

# FontTools drops an optional Windows font metadata table during subsetting.
# Embedded outlines and Unicode maps are retained.
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
PREVIEW = ROOT / 'publication-preview'
FIG.mkdir(exist_ok=True)
PREVIEW.mkdir(exist_ok=True)

BLUE = '#2266AE'
ORANGE = '#CC6035'
INK = '#172A3B'
MUTED = '#566674'
GRID = '#DEE5EB'
PALE = '#F1F5F9'
TEAL = '#247368'
WIDTH, HEIGHT = 595.28, 841.89
LEFT, RIGHT = 48, 547.28
CW = RIGHT - LEFT
COL = (CW - 24) / 2

FONTDIR = Path('C:/Windows/Fonts')
FALLBACK = Path(matplotlib.get_data_path()) / 'fonts/ttf'
FONTFILES = {
    'body': FONTDIR / 'georgia.ttf',
    'italic': FONTDIR / 'georgiai.ttf',
    'sans': FONTDIR / 'segoeui.ttf',
    'bold': FONTDIR / 'segoeuib.ttf',
}
for key in FONTFILES:
    if not FONTFILES[key].exists():
        FONTFILES[key] = FALLBACK / ('DejaVuSerif.ttf' if key in ('body', 'italic') else 'DejaVuSans-Bold.ttf' if key == 'bold' else 'DejaVuSans.ttf')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.labelcolor': MUTED, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.edgecolor': GRID,
    'axes.linewidth': .7, 'savefig.facecolor': 'white',
})

ROWS = {r['run']: r for r in csv.DictReader((ROOT / 'results/runs_index.csv').open(encoding='utf-8'))}
MANIFEST = {}

def result(label, names):
    vals = [Decimal(ROWS[n]['gen_ppl']) for n in names]
    mean = sum(vals) / len(vals)
    MANIFEST[label] = {'mean_gen_ppl': str(mean), 'runs': names, 'values': [str(v) for v in vals]}
    return float(mean), [float(v) for v in vals]

def names(prefix, seeds=(42, 43)):
    return [f'{prefix}_s{s}' for s in seeds]

def fmt(x, digits=2):
    return str(Decimal(str(x)).quantize(Decimal('1').scaleb(-digits), rounding=ROUND_HALF_UP))

best_suffix = '_51200_400x128_c50_lr1e-3_flex'
TOKEN = []
for label, prefix, seeds, note in [
    ('Random init · 2.4k', 'gen-mytok12std-2400x256', (42,43), '2 generator seeds'),
    ('Data init · 2.4k', 'gen-mytok12std-dati-2400x256', (42,43), '2 generator seeds'),
    ('+ Early restart · 2.4k', 'gen-mytok12std-dati-rv-2400x256', (42,43), '2 generator seeds'),
    ('R12 · 4.8k steps', 'gen-mytok12std-dati-rv-4800x256', (42,), '1 generator / tokenizer seed'),
]:
    mean, vals = result(label, names(prefix + best_suffix, seeds))
    TOKEN.append((label, mean, vals, note, BLUE))
long_names = names('gen-mytok12std-dati-rv-38400x256' + best_suffix, (42,)) + names('gen-mytok12std-dati-rv-38400x256-ts43' + best_suffix, (42,))
m, v = result('R12 · 38.4k steps', long_names)
TOKEN.append(('R12 · 38.4k steps', m, v, '2 tokenizer seeds; generator seed 42', BLUE))
m, v = result('Official tokenizer', names('gen-tokuff' + best_suffix))
TOKEN.append(('Official tokenizer', m, v, '2 generator seeds', ORANGE))

RECIPE = []
for label, suf in [
    ('Reconstructed recipe', 'lin75-25_lr3e-4'),
    ('Constant corruption 0.5', 'c50_lr3e-4'),
    ('+ FlexAttention', 'c50_lr3e-4_flex'),
    ('+ Learning rate 1e-3', 'c50_lr1e-3_flex'),
]:
    m, v = result('recipe: ' + label, names('gen-tokuff_51200_400x128_' + suf))
    RECIPE.append((label, m, v))

BUDGET = []
for exposure, steps in [(51200,400), (102400,800), (409600,3200)]:
    suffix = f'_{exposure}_{steps}x128_c50_lr1e-3_flex'
    o, ov = result(f'budget official {exposure}', names('gen-tokuff' + suffix))
    t, tv = result(f'budget own {exposure}', names('gen-mytok12std-dati-rv-2400x256' + suffix, (42,) if exposure == 102400 else (42,43)))
    BUDGET.append((exposure,o,t,ov,tv))

TRAINING_X = [4800, 9600, 19200, 38400]
TRAINING_Y = [TOKEN[3][1]]
for step in (9600, 19200):
    m, _ = result(f'long tokenizer snapshot {step}', names(f'gen-mytok12std-dati-rv-38400x256-p{step}' + best_suffix, (42,)))
    TRAINING_Y.append(m)
TRAINING_Y.append(TOKEN[4][2][0])
LONG_SECOND_SEED = TOKEN[4][2][1]

def axis_style(ax, axis='x'):
    ax.tick_params(length=0, pad=8, labelsize=9)
    ax.grid(axis=axis, color=GRID, linewidth=.65)
    ax.set_axisbelow(True)

def token_chart(fig, rect):
    ax = fig.add_axes(rect)
    axis_style(ax)
    for i,(label,mean,vals,note,color) in enumerate(TOKEN):
        yy = 5-i
        ax.barh(yy, mean, height=.52, color=color, alpha=.94)
        jit = [0] if len(vals) == 1 else [-.09,.09]
        ax.scatter(vals, [yy+j for j in jit], s=17, facecolors='white', edgecolors=INK, linewidths=.65, zorder=4)
        ax.text(max(vals)+4,yy,fmt(mean),va='center',fontsize=10,color=INK,weight='bold')
    ax.set_yticks(range(5,-1,-1), [x[0] for x in TOKEN])
    ax.set_xlim(0,255)
    ax.set_xticks([0,50,100,150,200,250])
    ax.set_ylim(-.6,5.6)
    ax.set_xlabel('Generative perplexity · lower is better',fontsize=9,labelpad=10)
    return ax

def recipe_chart(fig, rect):
    ax = fig.add_axes(rect)
    axis_style(ax)
    for i,(label,mean,vals) in enumerate(RECIPE):
        yy=3-i
        ax.barh(yy,mean,height=.52,color=BLUE if i==3 else '#82A6CD')
        ax.scatter(vals,[yy-.075, yy+.075],s=18,facecolors='white',edgecolors=INK,linewidths=.65,zorder=4)
        delta = '' if i == 0 else f'  ({(mean/RECIPE[0][1]-1)*100:.1f}%)'
        ax.text(max(vals)+4, yy, fmt(mean)+delta, va='center',fontsize=9.3, weight='bold' if i==3 else 'normal')
    ax.set_yticks([3,2,1,0], [x[0] for x in RECIPE])
    ax.set_xlim(0,195)
    ax.set_xticks([0,50,100,150])
    ax.set_ylim(-.6,3.6)
    ax.set_xlabel('Generative perplexity · lower is better',fontsize=9,labelpad=10)
    return ax

def budget_chart(fig, rect):
    ax=fig.add_axes(rect)
    axis_style(ax, 'y')
    x=[r[0] for r in BUDGET]
    for col,color,label,offset in [(2,BLUE,'Our R12 tokenizer · 2,400 steps',7),(1,ORANGE,'Official tokenizer',-9)]:
        y=[r[col] for r in BUDGET]
        ax.plot(x,y,'o-',color=color,lw=1.8,ms=5,label=label)
        for xx,yy in zip(x,y): ax.annotate(fmt(yy),(xx,yy),xytext=(0,offset),textcoords='offset points',ha='center',va='bottom' if offset>0 else 'top',fontsize=9,color=color,weight='bold')
    for xx,o,t,*_ in BUDGET:
        ax.text(xx,(o+t)/2, f'{t/o:.2f}×',ha='center',va='center',fontsize=9,color=MUTED,bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.set_xscale('log'); ax.set_xticks(x,['51,200','102,400','409,600']); ax.minorticks_off()
    ax.set_xlim(42000,510000); ax.set_ylim(0,135)
    ax.set_ylabel('Generative perplexity',fontsize=9)
    ax.set_xlabel('Generator stories seen · log scale',fontsize=9,labelpad=10)
    ax.legend(loc='upper right',fontsize=8.4,frameon=False)
    return ax

def training_chart(fig, rect):
    ax=fig.add_axes(rect)
    axis_style(ax,'y')
    x=TRAINING_X
    y=TRAINING_Y
    ax.scatter([x[0]],[y[0]],s=27,color=BLUE,zorder=4,label='Separate 4,800-step run')
    ax.plot(x[1:],y[1:],'o-',lw=1.7,ms=5,color=BLUE,label='Long run: tokenizer seed 42')
    ax.scatter([38400],[LONG_SECOND_SEED],s=36,marker='D',facecolors='white',edgecolors=BLUE,zorder=5,label='Long run: tokenizer seed 43')
    for xx,yy in zip(x,y):
        ax.annotate(fmt(yy),(xx,yy),xytext=(-10,-14) if xx==38400 else (0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.annotate(fmt(LONG_SECOND_SEED),(38400,LONG_SECOND_SEED),xytext=(8,6),textcoords='offset points',fontsize=9)
    ax.axhline(TOKEN[5][1],color=ORANGE,lw=1.1,ls=(0,(4,3)))
    ax.text(5200,68,f'Official tokenizer: {fmt(TOKEN[5][1])}',color=ORANGE,fontsize=8.5)
    ax.set_xscale('log',base=2); ax.set_xticks(x,['4,800','9,600','19,200','38,400']);ax.minorticks_off()
    ax.set_xlim(4000,51000); ax.set_ylim(62,117)
    ax.set_xlabel('Tokenizer training steps · log scale',fontsize=9,labelpad=10)
    ax.set_ylabel('Generative perplexity',fontsize=9)
    ax.legend(loc='upper right',fontsize=8.2,frameon=False)
    return ax

def save_standalone(name, fun, height, rect):
    fig=plt.figure(figsize=(8.4,height),facecolor='white')
    fun(fig,rect)
    for suffix in ('png','svg','pdf'):
        fig.savefig(FIG/f'{name}.{suffix}',dpi=220)
    plt.close(fig)

save_standalone('tokenizer_progress',token_chart,3.55,[.29,.17,.68,.79])
save_standalone('generator_recipe',recipe_chart,2.85,[.32,.21,.65,.72])
save_standalone('generator_budget',budget_chart,3.8,[.10,.17,.87,.79])
save_standalone('tokenizer_training',training_chart,3.5,[.11,.18,.85,.77])
(HERE/'figure-data.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')

# Page composition: point-based typesetting with measured line wrapping.
TTP = TextToPath()
TEXT_RECORD=[]

def prop(key='body',size=10): return FontProperties(fname=str(FONTFILES[key]),size=size)
def text_width(s,key,size): return TTP.get_text_width_height_descent(s,prop(key,size),False)[0]

class Page:
    def __init__(self, number, section, title=None, subtitle=None):
        self.number=number
        self.fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor='white')
        self.line(LEFT,HEIGHT-41,RIGHT,HEIGHT-41,GRID,.7)
        self.txt('NCM  /  COMPUTE-CONSTRAINED TRAINING',LEFT,HEIGHT-26,8,'bold',BLUE)
        self.txt(section.upper(),RIGHT,HEIGHT-26,7.5,'sans',MUTED,align='right')
        self.line(LEFT,43,RIGHT,43,GRID,.7)
        self.txt('Gianluca Meola  ·  Research note v0.1  ·  November 2026',LEFT,29,7.5,'sans',MUTED)
        self.txt(f'{number:02d} / 08',RIGHT,29,8,'bold',BLUE,align='right')
        self.y=HEIGHT-74
        if title:
            self.y=self.para(title,self.y,size=24,leading=29,key='bold',gap=12)
        if subtitle:
            self.y=self.para(subtitle,self.y,size=10,leading=14,key='sans',color=MUTED,gap=20)
    def txt(self,s,x,y,size=10,key='body',color=INK,align='left',url=None):
        t=self.fig.text(x/WIDTH,y/HEIGHT,s,fontproperties=prop(key,size),color=color,va='top',ha=align,parse_math=False,url=url)
        TEXT_RECORD.append({'page':self.number,'x':round(x,2),'y':round(y,2),'text':s})
        return t
    def line(self,x1,y1,x2,y2,color=GRID,lw=.7):
        self.fig.lines.append(plt.Line2D([x1/WIDTH,x2/WIDTH],[y1/HEIGHT,y2/HEIGHT],transform=self.fig.transFigure,color=color,lw=lw))
    def box(self,x,y,w,h,fc=PALE,ec=None):
        self.fig.patches.append(Rectangle((x/WIDTH,(y-h)/HEIGHT),w/WIDTH,h/HEIGHT,transform=self.fig.transFigure,facecolor=fc,edgecolor=ec,linewidth=.6))
    def para(self,s,y=None,x=LEFT,w=CW,size=10,leading=14.6,key='body',color=INK,gap=10):
        if y is None: y=self.y
        if key in ('body', 'italic'):
            s=s.replace('→', 'to')
        lines=[]
        for paragraph in s.split('\n'):
            line=''
            for word in paragraph.split():
                trial=(line+' '+word).strip()
                if line and text_width(trial,key,size)>w:
                    lines.append(line);line=word
                else:line=trial
            lines.append(line)
        for line in lines:
            if y-leading<51: raise ValueError(f'Page {self.number} overflow: {s[:80]} at y={y}')
            self.txt(line,x,y,size,key,color)
            y-=leading
        return y-gap
    def head(self,s,y=None,x=LEFT,w=CW,size=13):
        return self.para(s,self.y if y is None else y,x,w,size,17,'bold',gap=8)
    def caption(self,s,y,x=LEFT,w=CW):
        return self.para(s,y,x,w,8.5,11.7,'sans',MUTED,10)
    def callout(self,label,s,y,x=LEFT,w=CW):
        # Text is drawn before the background; lower z-order keeps the fill behind it.
        y2=self.para(label,y-12,x+13,w-26,9,12,'bold',BLUE,7)
        y2=self.para(s,y2,x+13,w-26,9.4,13.5,'sans',INK,13)
        self.box(x,y,w,y-y2)
        self.line(x,y,x,y2,BLUE,2)
        return y2-15
    def chart(self,fn,y,height,leftpad=0,rightpad=0):
        # Axes position includes room for labels and axis captions outside the axes.
        fn(self.fig,[(LEFT+leftpad)/WIDTH,(y-height)/HEIGHT,(CW-leftpad-rightpad)/WIDTH,(height-14)/HEIGHT])
        return y-height-48
    def table(self,headers,rows,widths,y,size=9,leading=12):
        assert abs(sum(widths)-CW)<1
        x=LEFT
        headheights=[]
        for h,w in zip(headers,widths):
            headheights.append(self.para(h,y-10,x+7,w-14,8.2,11,'bold',BLUE,0));x+=w
        yy=min(headheights)-9
        self.box(LEFT,y,CW,y-yy,PALE)
        for row in rows:
            ytop=yy;x=LEFT;ends=[]
            for val,w in zip(row,widths):
                ends.append(self.para(str(val),ytop-9,x+7,w-14,size,leading,'sans',gap=0));x+=w
            yy=min(ends)-9
            self.line(LEFT,yy,RIGHT,yy,GRID,.5)
        return yy-14
    def save(self,pdf):
        pdf.savefig(self.fig)
        self.fig.savefig(PREVIEW/f'page-{self.number:02d}.png',dpi=130)
        plt.close(self.fig)

PDF=HERE/'ncm-12gb.pdf'
metadata={'Title':'Training a Next-Concept Model on a single 12 GB GPU','Author':'Gianluca Meola','Subject':'Short-budget NCM training, codebook initialisation and the remaining tokenizer gap','Keywords':'NCM; TinyStories; VQ-VAE; generative perplexity; reproducibility','Creator':'Matplotlib; editable source in paper/build_paper.py','CreationDate':datetime(2026,11,20,tzinfo=timezone.utc)}

with PdfPages(PDF, metadata=metadata) as pdf:
    p=Page(1,'Research note')
    y=p.para('Training a Next-Concept Model\non a single 12 GB GPU',p.y,size=29,leading=35,key='bold',gap=14)
    y=p.para('Practical gains at small budgets, and a tokenizer gap that remains',y,size=12,leading=17,key='sans',color=MUTED,gap=16)
    p.txt('Gianluca Meola',LEFT,y,11,'bold'); y-=19
    p.txt('gianluca.meola89@gmail.com',LEFT,y,9.5,'sans',BLUE,url='mailto:gianluca.meola89@gmail.com')
    p.txt('20 November 2026  ·  v0.1  ·  Work in progress',RIGHT,y,8.5,'sans',MUTED,align='right'); y-=29
    y=p.head('Abstract',y,size=11)
    y=p.para('Next-Concept Models generate text through a hierarchy of discrete codes. I train both the tokenizer and generator on one RTX 3080 Ti (12 GB), using TinyStories and budgets far below the original paper. Data-based codebook initialisation lowers generative perplexity by 48% in the tested tokenizer comparison. A retuned generator recipe lowers it by 56% on the official tokenizer, with most of the gain coming from the learning rate. Longer tokenizer training helps, but a 1.39× gap to the official tokenizer remains. The study reports reproducible runs, unsuccessful interventions and limits on causal interpretation.',y,size=10,leading=14.4,gap=17)
    # Compact takeaways: each number has a concrete, qualified label.
    cards=[('−48%','Data-based codebook init','222.78 → 115.38'),('−56%','Short-budget recipe','150.67 → 66.26'),('1.39×','Remaining tokenizer gap','92.35 / 66.26')]
    cardw=(CW-16)/3
    for i,(big,label,small) in enumerate(cards):
        x=LEFT+i*(cardw+8);p.box(x,y,cardw,80)
        p.txt(big,x+11,y-10,24,'bold',BLUE)
        p.txt(label,x+11,y-44,8.5,'bold')
        p.txt(small,x+11,y-60,8.5,'sans',MUTED)
    y-=98
    y=p.head('A large gain, with an unfinished gap',y,size=12)
    y=p.chart(token_chart,y-1,163,leftpad=132,rightpad=7)
    y=p.caption('Figure 1. Same generator recipe and 51,200-story budget, different tokenizers. Bars show means; white markers are individual training runs. The 38.4k row varies tokenizer seed with generator seed fixed; other paired rows vary generator seed. Gen. PPL is lower-is-better, not a percentage measure of general text quality.',y)
    y=p.para('This is a study of constrained training, not a full reproduction: the longest tokenizer sees 9.83M stories versus 122.88M in the paper; the longest generator sees 0.41M versus 102.4M.',y,size=9,leading=12.8,key='sans',color=MUTED,gap=0)
    p.save(pdf)

    p=Page(2,'Protocol','1  Experimental setup','The unit of comparison is a training recipe at a stated data exposure, with tokenizer and generator budgets kept separate.')
    y=p.y
    # Simple vector pipeline with distinct stages and explicit generation semantics.
    bw=(CW-36)/4
    for i,(a,b) in enumerate([('128 BPE tokens','TinyStories input'),('VQ-VAE tokenizer','12 code levels'),('Level-wise generator','Predict levels 1–11'),('Tokenizer decoder','Output text')]):
        x=LEFT+i*(bw+12);p.box(x,y,bw,59)
        p.para(a,y-11,x+8,bw-16,9,12,'bold',gap=0)
        p.para(b,y-36,x+8,bw-16,8,10,'sans',MUTED,0)
        if i<3:p.txt('→',x+bw+2,y-23,12,'sans',BLUE)
    y-=75
    y=p.caption('At generation time, level 0 is sampled from stored empirical frequencies. Eleven generator passes produce levels 1–11 for a 128-token story [1, 2]. The pass count is not a measured wall-clock speedup in this study.',y)
    y=p.table(['Quantity','This study','Original paper'],[
        ['Tokenizer parameters','36.4M','Same base architecture'],
        ['Generator parameters','88.6M at 12 levels','Same base architecture'],
        ['Longest tokenizer exposure','9,830,400 stories','122,880,000 stories'],
        ['Longest generator exposure','409,600 stories','102,400,000 stories'],
        ['Main generator comparison','400 updates × 128 stories','Not budget-matched'],
    ],[164,168,CW-332],y)
    x2=LEFT+COL+24
    yl=p.head('Evaluation protocol',y,w=COL)
    yl=p.para('Each trained generator produces 1,280 unconditional stories: 256 samples for each sampling seed 42, 0, 1, 2 and 3. Temperature is 1, with no top-k or top-p truncation. The authors’ evaluation scripts score the samples with GPT-2 large.',yl,w=COL)
    yl=p.para('Generative perplexity (Gen. PPL) is the primary metric. Generated-text entropy must not fall more than 0.1 below the comparison reference; this guards against some repetitive outputs. Validation uses the first 4,096 held-out stories.',yl,w=COL)
    yl=p.head('Two different kinds of seeds',yl,w=COL,size=11)
    yl=p.para('Training seeds vary initialisation while data order is fixed. Sampling seeds vary the outputs of one trained model. Standard deviations across sampling seeds in the run report are not confidence intervals across training runs.',yl,w=COL,size=9.7,leading=14)
    yr=p.head('Decision rules and scope',y,x=x2,w=COL)
    yr=p.para('Decision rules were written before the corresponding experiments. Later cycles used 6.7% between single runs and 4.7% between two-seed means, based on 2.4% training-seed noise measured over five pairs at 51,200 stories. Early cycles used a fixed five-PPL-point threshold.',yr,x=x2,w=COL)
    yr=p.para('These are operational thresholds, not formal confidence intervals or corrections for a search over many recipes. Most tokenizers were trained once; only the longest setting has two tokenizer seeds.',yr,x=x2,w=COL)
    yr=p.head('Compute and data exposure',yr,x=x2,w=COL,size=11)
    yr=p.para('Stories seen count examples processed, including repeats. Equal exposure does not mean equal optimiser updates or FLOPs. All results concern TinyStories, 128-token sequences and the tested model sizes; no larger-language-model distillation is used in this study.',yr,x=x2,w=COL,size=9.7,leading=14)
    p.save(pdf)

    p=Page(3,'Generator','2  The short-budget recipe','A higher learning rate provides most of the measured improvement on the frozen official tokenizer.')
    y=p.y
    y=p.chart(recipe_chart,y,150,leftpad=145,rightpad=7)
    y=p.caption('Figure 2. Successive changes at 51,200 stories, 400 updates × 128 stories. Bars are means of generator seeds 42 and 43; markers show individual runs. Percentages are relative to the reconstructed baseline, not the preceding row.',y)
    y=p.table(['Generator setting','51,200 stories','102,400 stories'],[
        ['Reconstructed paper recipe','150.67','74.82'],
        ['Constant corruption + Flex + lr 1e-3','66.26','43.84'],
        ['Relative Gen. PPL reduction','56.0%','41.4%'],
    ],[266,116,CW-382],y)
    x2=LEFT+COL+24
    yl=p.head('What changes',y,w=COL)
    yl=p.para('The baseline uses corruption decreasing from 0.75 to 0.25 across levels and lr 3e-4. Constant corruption at 0.5 lowers PPL by 8.5%; adding FlexAttention gives an 11.7% reduction from baseline. Raising lr to 1e-3 produces the largest change.',yl,w=COL)
    yl=p.para('The combined recipe retains a substantial gain at 102,400 stories on two generator seeds. Its performance at the full paper budget is untested.',yl,w=COL)
    yr=p.head('What this does not isolate',y,x=x2,w=COL)
    yr=p.para('The FlexAttention path omits attention dropout used by the baseline. Its effect mixes implementation and regularisation changes. The improvement belongs to the combined recipe, not to the attention kernel alone.',yr,x=x2,w=COL)
    yr=p.para('Official training code was unavailable in the checked sources, and the paper omits some training details. The baseline is a reconstruction, not a verified reproduction of the authors’ run.',yr,x=x2,w=COL)
    y=min(yl,yr)-2
    y=p.callout('Equal stories do not imply equal optimisation', 'At the same 51,200 stories, 800 updates × 64 give 83.16 for the baseline, versus 150.67 at 400 × 128. The alternative has one seed: the 56% result belongs to the stated short-budget comparison.',y)
    p.save(pdf)

    p=Page(4,'Tokenizer','3  Initialisation, then training','A tokenizer is evaluated by the text generated from its codes, not by reconstruction loss alone.')
    y=p.y
    y=p.table(['Tokenizer at 2,400 steps','Gen. PPL','Perfect stories','L0 codes used'],[
        ['Random codebook initialisation','222.78','59.6%','14'],
        ['Initialisation from data (R1)','115.38','51.8%','17'],
        ['R1 + early dead-code restart (R12)','108.17','39.9%','22'],
    ],[234,76,99,CW-409],y,size=8.8)
    y=p.caption('PPL means use two generator seeds per frozen tokenizer; each tokenizer has one training seed. Reconstruction and distinct level-0 code counts use 4,096 validation stories. Early restarts stop after step 1,200.',y)
    y=p.para('Data initialisation produces the clearest tokenizer gain: −48.2% PPL with the same generator recipe. Early restart adds −6.2%, but the isolated restart effect still needs a second tokenizer training seed. Early-only restart is not a new proposal: upstream PR #27 and issue #16 predate this study [3, 4].',y,gap=12)
    y=p.chart(training_chart,y,170,leftpad=37,rightpad=9)
    y=p.caption('Figure 3. Generator budget fixed at 51,200 stories. The 9,600- and 19,200-step points are snapshots on one long, decaying learning-rate schedule. The unconnected 4,800-step point is a separate run; the second long-run tokenizer scores 93.23 with generator seed 42.',y)
    x2=LEFT+COL+24
    yl=p.head('Longer training helps',y,w=COL,size=12)
    yl=p.para('At 38,400 steps, two tokenizer seeds give 91.46 and 93.23 (mean 92.35), 10.9% below the single 4,800-step baseline of 103.61. Relative to 66.26 on the official tokenizer, the ratio remains 1.39×. The official tokenizer saw 12.5× more stories.',yl,w=COL,size=9.5,leading=13.5)
    yr=p.head('Reconstruction is an imperfect guide',y,x=x2,w=COL,size=12)
    yr=p.para('Generation improves even as perfect reconstruction falls. Across ten own tokenizer evaluations, code usage has Spearman correlation −0.90 with PPL (−0.83 without two snapshots), versus −0.48 for perfect-story reconstruction. This small, post-hoc association has exceptions and does not replace a generator evaluation.',yr,x=x2,w=COL,size=9.5,leading=13.5)
    p.save(pdf)

    p=Page(5,'Remaining gap','4  A gap under longer generator training','This sweep uses the 2,400-step R12 tokenizer. It is a separate experiment from the 38,400-step result on the previous page.')
    y=p.y
    y=p.chart(budget_chart,y,201,leftpad=37,rightpad=8)
    y=p.caption('Figure 4. Both tokenizers are frozen. Two generator seeds at 51,200 and 409,600 stories; at 102,400, two on the official tokenizer and one on R12. Separately, the authors’ full-budget model scores 16.58 here versus 15.41 in the paper (+7.6%, unexplained); those are not matched-budget results.',y)
    y=p.para('Increasing generator exposure eightfold improves both models: 66.26 → 25.88 on the official tokenizer and 108.17 → 42.00 on R12. The ratio is 1.63×, 1.50× and 1.62× across the three budgets. This recipe and exposure range do not close the gap for this tokenizer pair.',y,gap=13)
    y=p.table(['Intervention','Observed Gen. PPL','What remains uncertain'],[
        ['Tokenizer batch, equal 1.23M stories','256: 103.61; 1,024: 98.29; 4,096: 100.36','One seed each; long training with a large batch untested.'],
        ['Remove level-0 normalisation','115.28 vs 103.61','Worse in this setting; other normalisation designs untested.'],
        ['Official encoding in fp64','66.53 vs 66.26','No material effect detected on two seeds at this budget.'],
    ],[156,150,CW-306],y,size=8.6,leading=11.6)
    y=p.callout('Evidence narrows the search; it does not exclude whole causes', 'The 1.39× gap from the longer tokenizer has not received the same generator-budget sweep. Similar validation cross-entropies across different tokenizers do not establish equal prediction quality, because their target code distributions differ.',y)
    p.save(pdf)

    p=Page(6,'Diagnostics','5  Level 0: an observation, not a cause','The coarsest code plays different roles in reconstruction and in the generator’s conditioning.')
    y=p.y
    y=p.table(['Diagnostic','Official tokenizer','Tokenizer trained here'],[
        ['Decoded tokens changed by an L0 swap','0.07%','4.0–5.5% in R12 batch-256 tokenizers at 4,800+ steps'],
        ['Generator loss rise after an L0 swap','+0.34 to +0.53 nats','About +0.25 to +0.29 nats in tested models'],
        ['Extra token errors: 10% corruption, levels 1–11','1.04%','1.14% at 38,400 steps'],
    ],[226,108,CW-334],y,size=8.8,leading=12)
    y=p.para('The official decoder is nearly insensitive to swapping level 0, yet the generator relies on that code: its loss rises when level 0 comes from a different story. In several tokenizers trained here, the decoder is more sensitive. This makes the role of level 0 a useful open question, not a demonstrated explanation of the PPL gap.',y,gap=13)
    y=p.callout('A counterexample matters', 'The tokenizer trained at batch 4,096 also has an almost inert level 0: only 0.05% of decoded tokens change when it is swapped. Its Gen. PPL is still 100.36. An inert level 0 alone is therefore insufficient to reproduce the official tokenizer’s result.',y)
    y=p.head('One held-out story, decoded from progressively more levels',y,size=11)
    data=json.loads((ROOT/'figures/coarse_to_fine.json').read_text(encoding='utf-8'))
    # A compact static comparison replaces the animated GIF in a printed document.
    cols=[('Official tokenizer','official',ORANGE),('R12 · 38,400 steps','ours (38,400 steps)',BLUE)]
    x2=LEFT+COL+24
    for x,(label,key,color) in zip([LEFT,x2],cols):
        p.txt(label,x,y,9.5,'bold',color)
    y-=23
    for idx,label in [(0,'LEVEL 0 ONLY'),(5,'LEVELS 0–5'),(11,'LEVELS 0–11')]:
        row_bottom=[]
        for x,(_,key,color) in zip([LEFT,x2],cols):
            frame=data['frames'][key][idx]
            p.txt(f'{label}  ·  {frame["token_accuracy"]:.0%} tokens correct',x,y,7.5,'bold',color)
            excerpt=' '.join(frame['text'].split())[:110].rsplit(' ',1)[0]+'…'
            row_bottom.append(p.para(excerpt,y-15,x,COL,8.7,12,'body',gap=10))
        y=min(row_bottom)-4
    y=p.caption('Illustration: validation story index 1, from figures/coarse_to_fine.json. Text excerpts are truncated. This is reconstruction from encoded reference text, not free generation. Accuracy describes this one example, not an aggregate.',y)
    y=p.para('Random code substitutions probe decoder sensitivity; they do not reproduce the generator’s actual error distribution. Similar aggregate damage at levels 1–11 cannot rule out per-level effects or correlated prediction errors.',y,size=9,leading=12.5,key='sans',color=MUTED)
    p.save(pdf)

    p=Page(7,'Discussion & artifacts','6  What the study supports','A practical training record with measurable gains; the full-budget explanation remains open.')
    y=p.y;x2=LEFT+COL+24
    yl=p.head('Supported conclusions',y,w=COL,size=12)
    yl=p.para('Data-based codebook initialisation is a strong improvement in the tested short-budget comparison. The generator learns substantially faster with the retuned short-run recipe. Extending tokenizer training improves downstream PPL, but does not match the official tokenizer.',yl,w=COL)
    yl=p.para('Short generator runs can reveal a large difference between a fixed pair of tokenizers. They are less reliable for nearby choices: the 4,800- versus 2,400-step ranking reverses when generator exposure increases from 51,200 to 102,400 stories, with differences within the study’s noise thresholds.',yl,w=COL)
    yr=p.head('Limits and next discriminating tests',y,x=x2,w=COL,size=12)
    yr=p.para('The evidence covers one small English dataset, short sequences, few training seeds and one automatic text-quality metric. It does not establish competitive general language ability, a new state of the art, or full reproduction of the original NCM result.',yr,x=x2,w=COL)
    yr=p.para('The next useful controls are a generator-budget sweep on the 38,400-step tokenizer, an independent tokenizer seed for the isolated restart effect, and longer tokenizer training with matched batch and schedule choices. A decoder-inert level 0 is a possible experiment, not a validated remedy.',yr,x=x2,w=COL)
    y=min(yl,yr)-2
    y=p.head('Prepared checkpoints and required pairings',y,size=12)
    y=p.table(['Checkpoint','Result / pairing'],[
        ['R12 tokenizer, 38,400 steps, seed 42','91.46 with a 51,200-story generator; that generator is not in the prepared model package.'],
        ['R12 tokenizer, 2,400 steps, seed 42','Required by the packaged generator on our tokenizer.'],
        ['Generator on R12, 409,600 stories, seed 42','42.76; pair only with the 2,400-step tokenizer.'],
        ['Generator on official tokenizer, same budget','26.04; official tokenizer is an upstream dependency.'],
    ],[236,CW-236],y,size=8.8,leading=12)
    y=p.para('Weights: huggingface.co/janlu89/ncm-12gb. Mean results in the figures must not be confused with individual checkpoint scores. Different tokenizer checkpoints learn different code spaces and are not interchangeable for a fixed generator.',y,size=9.5,leading=13.5,gap=13)
    y=p.head('Measured resource use',y,size=12)
    y=p.para('On an RTX 3080 Ti, a 51,200-story generator run takes roughly 10–12 minutes and peaks at 7.0 GiB allocated by PyTorch; 409,600 stories take 62–70 minutes. A 38,400-step tokenizer takes about 3 h 15 min. Host: Intel i5-6600K, Windows 10, PyTorch 2.14.1 / CUDA 12.6. These are local timings for the specified configurations, not a total project-cost estimate.',y,size=9.5,leading=13.5)
    p.save(pdf)

    p=Page(8,'References & reproducibility','7  Reproducibility and references','The narrative is a guide to the evidence; configurations, logs and evaluation outputs remain the detailed record.')
    y=p.y
    y=p.table(['Repository artifact','Purpose'],[
        ['README.md / REPRODUCING.md','Research overview; environment, upstream assets and exact training/evaluation commands.'],
        ['EXPERIMENTS.md','Chronological questions, decision rules, results and limitations for 16 experiments.'],
        ['results/RUNS.md / results/runs_index.csv','18 tokenizer runs, 77 evaluated generator runs, 5 interrupted or cancelled runs.'],
        ['results/runs/<run>/','Per-run configuration, logs, evaluation seeds and generated sample text.'],
        ['paper/build_paper.py / figure-data.json','Editable PDF source and exact run IDs behind the principal plots.'],
    ],[226,CW-226],y,size=8.9,leading=12)
    y=p.para('Training checkpoints support pause and resume. A deterministic short test matches uninterrupted training exactly; the reported research runs use the faster non-deterministic mode. Interrupted run folders are retained. The study does not rerun or overwrite model weights when building this document.',y,size=9.5,leading=13.5,gap=14)
    y=p.head('Primary references',y,size=12)
    refs=[
        ('[1] Kumbong, H., et al. (2026). Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation. arXiv:2610.14162.', 'arxiv.org/abs/2610.14162','https://arxiv.org/abs/2610.14162'),
        ('[2] Hazy Research. NCM: architecture code, evaluation scripts and released checkpoint references. Upstream revision is pinned in third_party/README.md.', 'github.com/HazyResearch/ncm','https://github.com/HazyResearch/ncm'),
        ('[3] HazyResearch/ncm, pull request #27. Prior early-restart stopping proposal and related codebook changes.', 'github.com/HazyResearch/ncm/pull/27','https://github.com/HazyResearch/ncm/pull/27'),
        ('[4] HazyResearch/ncm, issue #16. Discussion of code restarts and reconstruction versus generation.', 'github.com/HazyResearch/ncm/issues/16','https://github.com/HazyResearch/ncm/issues/16'),
        ('[5] Eldan, R., and Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? Dataset used here through the NCM authors’ pre-tokenised release.', 'huggingface.co/datasets/roneneldan/TinyStories','https://huggingface.co/datasets/roneneldan/TinyStories'),
    ]
    for desc,label,url in refs:
        y=p.para(desc,y,size=9,leading=12.5,key='sans',gap=3)
        p.txt(label,LEFT,y,8.4,'sans',BLUE,url=url);y-=21
    y=p.head('Author and publication status',y,size=11)
    y=p.para('Gianluca Meola · Independent research note · v0.1, 20 November 2026. Repository: github.com/janlu89/ncm-12gb. Code licensing and upstream attribution are recorded in LICENSE and third_party/.',y,size=9,leading=12.5,key='sans',gap=8)
    p.txt('Contact: gianluca.meola89@gmail.com',LEFT,y,10,'bold',BLUE,url='mailto:gianluca.meola89@gmail.com')
    p.save(pdf)

(HERE/'layout-text.json').write_text(json.dumps(TEXT_RECORD,ensure_ascii=False,indent=2),encoding='utf-8')

# A compact overview for reviewing the eight page PNG exports.
thumbs=[]
for i in range(1,9):
    im=Image.open(PREVIEW/f'page-{i:02d}.png').convert('RGB')
    im.thumbnail((298,422))
    tile=Image.new('RGB',(322,454),'#e8edf2')
    tile.paste(im,((322-im.width)//2,12))
    ImageDraw.Draw(tile).text((12,436),f'Page {i}',fill=INK)
    thumbs.append(tile)
sheet=Image.new('RGB',(322*4,454*2),'#e8edf2')
for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*322,(i//4)*454))
sheet.save(PREVIEW/'contact-sheet.png')
print(f'Built {PDF} ({PDF.stat().st_size:,} bytes), eight pages; figure values drawn from {len(MANIFEST)} groups in the run index.')
