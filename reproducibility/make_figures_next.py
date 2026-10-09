"""Publication figures from executed results; editable vector and 420-dpi assets.

The feasible-region plots depict the declared finite completion model (and its
continuous convex hull), not new biological observations. No generated artwork.
"""
from pathlib import Path
import json,sys,io,os,uuid
import fitz
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import patheffects
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures_next';OUT.mkdir(exist_ok=True)
F=pd.read_csv(ROOT/'completion_results/families.csv')
I=pd.read_csv(ROOT/'completion_results/interpretations.csv')
P=pd.read_csv(ROOT/'completion_results/pairs.csv')
U=pd.read_csv(ROOT/'completion_results/uniform_planning_summary.csv')
C=pd.read_csv(ROOT/'results/cohort_summary.csv')
PROTOCOL=json.loads((ROOT/'reproducibility/protocol.json').read_text())
GENES=PROTOCOL['genes'];COHORTS=[c.upper() for c in PROTOCOL['original_cohorts']+PROTOCOL['additional_cohorts']]
INK='#183346';TEAL='#147D8C';RED='#A83C56';GOLD='#B9852B';GRAY='#8995A0';PALE='#F0F4F6';BLUE='#3F679A'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'mathtext.fontset':'stix',
 'axes.labelsize':8,'axes.titlesize':8.8,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.linewidth':.55,'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,
 'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})


def save(fig,name):
    for ext in ['pdf','svg','png']:
        buffer=io.BytesIO()
        fig.savefig(buffer,format=ext,dpi=420,bbox_inches='tight',pad_inches=.055)
        content=buffer.getvalue()
        if ext=='pdf':
            assert b'%%EOF' in content[-100:]
            doc=fitz.open(stream=content,filetype='pdf');assert len(doc)==1;doc.close()
        elif ext=='png':
            Image.open(io.BytesIO(content)).verify()
        else:ET.fromstring(content)
        temporary=OUT/(name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_bytes(content);os.replace(temporary,OUT/(name+'.'+ext))
    plt.close(fig)


def label(ax,letter,title):
    ax.text(0,1.035,letter,transform=ax.transAxes,fontweight='bold',fontfamily='DejaVu Sans',fontsize=9.5,va='bottom')
    ax.text(.075,1.035,title,transform=ax.transAxes,fontsize=8.6,va='bottom')


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.55)


def architecture():
    fig=plt.figure(figsize=(7.15,2.48));ax=fig.add_axes([.005,.015,.99,.97]);ax.set(xlim=(0,10),ylim=(0,3.25));ax.axis('off')
    def box(x,y,w,h,title,body,color=INK,fill='white',title_size=8.7):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.055',lw=.8,ec=color,fc=fill))
        ax.text(x+.11,y+h-.13,title,fontsize=title_size,fontweight='bold',ha='left',va='top',color=color)
        ax.text(x+.11,y+h-.49,body,fontsize=7.8,ha='left',va='top',linespacing=1.5)
    def arrow(x1,y1,x2,y2,color=INK):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',lw=.8,mutation_scale=8,color=color))
    ax.add_patch(Rectangle((.05,.55),6.17,2.62,ec=GRAY,fc='white',lw=.7))
    ax.text(.16,3.13,'EXECUTION + SHARED OBSERVATION MODEL',fontsize=7.5,fontweight='bold',va='top',color=GRAY)
    box(.22,1.56,1.64,1.13,'Specifications',r'$i\in\{0,1\}^{4}$'+'\nunit · assay\ncontext · class')
    box(2.17,1.56,1.63,1.13,'Eligible counts',r'$n_i,\; k_i^{\rm obs}$'+'\nSQL = pandas\ngene-panel adapter')
    box(4.12,1.56,1.86,1.13,'Patient blocks',r'$z_r\in\mathcal{Z}_r$'+'\nshared observations\npatient OR / sum')
    arrow(1.89,2.11,2.13,2.11);arrow(3.83,2.11,4.08,2.11)
    box(.22,.73,5.76,.59,'Sharp support: '+r'$\max_z\!\sum_r h_r(z_r)=\sum_r\!\max_{z_r}h_r(z_r)$','',TEAL,PALE,8.7)
    arrow(5.05,1.52,5.05,1.34,TEAL)
    box(6.57,1.56,3.15,1.13,'Violating-pair graph',r'$E_{ij}=\mathbf{1}[D_{ij}>\tau\;\vee\;A_{ij}>\beta]$'+'\none world shared by both queries\nstop when no violating edge remains',TEAL,PALE)
    arrow(6.05,1.03,6.38,1.03,TEAL);arrow(6.38,1.03,6.38,2.12,TEAL);arrow(6.38,2.12,6.53,2.12,TEAL)
    box(6.57,.73,3.15,.60,'Cost-optimal clarification','',INK,'white',8.4)
    arrow(8.15,1.52,8.15,1.36)
    ax.text(.22,.24,'Semantic guard: compare supported interpretations.',fontsize=8,color=TEAL)
    ax.text(5.1,.24,'Point guard: '+r'$(U_i-L_i)/2\leq\eta$'+'.',fontsize=8,color=RED)
    save(fig,'01_method')


def matrix():
    fig=plt.figure(figsize=(7.15,3.32));ax=fig.add_axes([.081,.253,.766,.662]);side=fig.add_axes([.875,.253,.104,.662])
    val=F.pivot(index='cohort',columns='gene',values='coupled_absolute_pp').reindex(index=COHORTS,columns=GENES).values
    im=ax.imshow(val,cmap='cividis',vmin=0,vmax=65,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(12),COHORTS,fontsize=7.5);ax.set_xticks(range(30),GENES,rotation=60,ha='right',fontsize=7)
    ax.tick_params(length=0,pad=3);ax.set_xticks(np.arange(-.5,30,1),minor=True);ax.set_yticks(np.arange(-.5,12,1),minor=True)
    ax.grid(which='minor',color='white',lw=.25,alpha=.6);ax.tick_params(which='minor',length=0)
    for y,cohort in enumerate(COHORTS):
        for x,gene in enumerate(GENES):
            r=F.loc[F.cohort.eq(cohort)&F.gene.eq(gene)].iloc[0]
            if not r.observed_sensitive and r.coupled_sensitive:
                ax.plot(x,y,marker='D',ms=3.2,mfc='white',mec=INK,mew=.5)
            elif not r.coupled_sensitive:
                ax.add_patch(Rectangle((x-.42,y-.41),.84,.82,lw=1.15,ec='white',fc='none'))
    ax.text(0,1.053,'a',transform=ax.transAxes,fontfamily='DejaVu Sans',fontweight='bold',fontsize=10)
    ax.text(.035,1.053,'Sharp completion-aware semantic diameter',transform=ax.transAxes,fontsize=9.1)
    coverage=C.set_index('cohort').reindex(COHORTS)
    missing=100*(coverage.samples-coverage.profiled_samples)/coverage.samples
    side.barh(np.arange(12),missing,color=GRAY,height=.73)
    side.set(ylim=(11.5,-.5),xlim=(0,26));side.set_yticks([]);side.set_xticks([0,10,20]);side.set_xlabel('% unprofiled',fontsize=7)
    side.spines[['top','right','left']].set_visible(False);side.tick_params(length=2)
    for y,v in enumerate(missing):side.text(v+.45,y,f'{v:.1f}',fontsize=6.5,va='center')
    side.text(0,1.053,'b',transform=side.transAxes,fontfamily='DejaVu Sans',fontweight='bold',fontsize=10)
    cbax=fig.add_axes([.085,.053,.38,.04]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,5,15,30,45,65])
    cb.ax.tick_params(labelsize=7,length=2);cb.set_label(r'$A(C)=100\max_{i,j\in C}\max_z|f_i(z)-f_j(z)|$  (percentage points)',fontsize=7.4,labelpad=3)
    fig.text(.51,.064,'◇  Recorded-only stable → completion sensitive\n□  Stable under both tolerances',fontsize=7.25,linespacing=1.7)
    save(fig,'02_matrix')


def geometry():
    fig,axs=plt.subplots(1,3,figsize=(7.15,3.0));fig.subplots_adjust(left=.063,right=.983,bottom=.36,top=.86,wspace=.43)
    # Exact KIRC VHL marginal endpoints from executed counts.
    vals=I[I.cohort.eq('kirc')&I.gene.eq('VHL')].set_index('index')
    a,b=axs[0],axs[1];L,U=100*vals.loc[4,['lower','upper']].astype(float)
    a.add_patch(Rectangle((L,L),U-L,U-L,fc=RED,alpha=.10,ec=RED,lw=.6))
    a.plot([L,U],[L,U],color=TEAL,lw=2);a.scatter([L,U],[L,U],s=12,c=TEAL,zorder=4)
    a.plot([L,U],[U,L],'x',color=RED,ms=6,mew=1.1)
    a.set(xlim=(L-2,U+2),ylim=(L-2,U+2),xlabel='Patient fraction (%)',ylabel='Sample fraction (%)')
    a.set_xticks([35,45,55]);a.set_yticks([35,45,55]);a.set_aspect('equal');clean(a);label(a,'a','Shared specimens')
    a.text(.05,.80,r'$A_{ij}=0$',transform=a.transAxes,color=TEAL,fontsize=9)
    a.text(.05,.68,'Box: 21.48 pp',transform=a.transAxes,color=RED,fontsize=7.6)
    a.text(.03,-.50,'KIRC · VHL · broad classes\nSame unit count; uncertain point value.',transform=a.transAxes,fontsize=7,linespacing=1.45)
    tL,tU=100*vals.loc[5,['lower','upper']].astype(float)
    x=np.linspace(L,U,160);y=np.linspace(tL,tU,160);X,Y=np.meshgrid(x,y)
    # The hull is triangle (L,tL), (U,tL), (U,tU); the lattice has b>=t.
    ok=(Y-tL<=X-L+1e-10);R=np.where(ok,(X-Y)/((X+Y)/2+1e-10),np.nan)
    b.pcolormesh(X,Y,R,cmap='cividis',shading='auto',vmin=0,vmax=1,rasterized=True)
    b.add_patch(Polygon([(L,tL),(U,tL),(U,tU)],closed=True,fill=False,ec=TEAL,lw=1.1))
    b.add_patch(Rectangle((L,tL),U-L,tU-tL,fill=False,ec=RED,lw=.7,ls='--'))
    b.plot(L,tU,'x',color=RED,ms=6,mew=1.1);b.text(L+1,tU-2,'Impossible',color=RED,fontsize=6.7)
    b.set(xlim=(L-2,U+2),ylim=(tL-2,tU+2),xlabel='Broad fraction (%)',ylabel='Restricted-class fraction (%)')
    b.set_xticks([35,45,55]);b.set_yticks([20,30,40]);b.set_aspect('equal');clean(b);label(b,'b','Nested event classes')
    cbax=b.inset_axes([.12,.69,.30,.035]); cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='cividis'),cax=cbax,orientation='horizontal',ticks=[0,1]);cb.ax.tick_params(labelsize=5.5,length=1,pad=1)
    b.text(.03,-.50,r'$z_j\in\{0,B,T\},\quad T\subseteq B$'+'\nColor: relative discrepancy (0–1).',transform=b.transAxes,fontsize=7,linespacing=1.45)
    c=axs[2];v=I[I.cohort.eq('ov')&I.gene.eq('BRCA1')].set_index('index');xl,xu=100*v.loc[4,['lower','upper']].astype(float);yl,yu=100*v.loc[6,['lower','upper']].astype(float)
    c.add_patch(Rectangle((xl,yl),xu-xl,yu-yl,fc=RED,alpha=.10,ec=RED,lw=.6))
    c.plot([xl,xu],[yl,yu],color=TEAL,lw=2);c.scatter([xl,xu],[yl,yu],s=12,c=TEAL,zorder=4)
    c.plot([xl,xu],[yu,yl],'x',color=RED,ms=6,mew=1.1)
    c.set(xlim=(0,16),ylim=(0,16),xlabel='All-specimen fraction (%)',ylabel='Primary-specimen fraction (%)');c.set_xticks([0,5,10,15]);c.set_yticks([0,5,10,15]);c.set_aspect('equal');clean(c);label(c,'c','Dependent queries')
    c.text(.06,.95,'Sharp: 0.047 pp',color=TEAL,transform=c.transAxes,fontsize=7.5)
    c.text(.06,.77,'Box: 10.645 pp',color=RED,transform=c.transAxes,fontsize=7.5)
    c.text(.03,-.50,'OV · BRCA1 · cohort denominator\nDependency persists beyond equivalence.',transform=c.transAxes,fontsize=7,linespacing=1.45)
    save(fig,'03_geometry')


def planning():
    fig,axs=plt.subplots(1,2,figsize=(7.15,2.95),gridspec_kw={'width_ratios':[1.16,1]});fig.subplots_adjust(left=.25,right=.98,bottom=.32,top=.86,wspace=.55)
    a=axs[0];u=U.set_index('certificate');c=u.loc['coupled']
    values=[u.loc['box','exact_expected'],u.loc['box_merged','exact_expected'],c.exact_expected,c.best_global_fixed,c.entropy_mean]
    names=['Independent box','Box + exact equivalence','Sharp coupled','Best global fixed¹','Entropy split¹']
    colors=[RED,GOLD,TEAL,INK,GRAY]
    for y,(v,n,col) in enumerate(zip(values,names,colors)):
        a.barh(y,v,color=col,height=.58);a.text(v+.045,y,f'{v:.3f}',va='center',fontsize=7.3)
    a.set(yticks=range(5),yticklabels=names,ylim=(4.6,-.65),xlim=(0,4.35),xlabel='Expected questions (uniform intents)');a.set_xticks([0,1,2,3,4]);clean(a);a.grid(axis='x',lw=.35,alpha=.25);a.set_axisbelow(True);label(a,'a','Certificates and policies')
    a.text(-.73,-.42,'¹ Uses the same sharp coupled stopping rule.',transform=a.transAxes,fontsize=6.8)
    b=axs[1];grid=pd.read_csv(ROOT/'completion_results/prior_cost_grid.csv')
    for k,col,l in [('equal',TEAL,'Equal costs'),('scope_expensive',GOLD,'Scope costs 3'),('eligibility_expensive',RED,'Assay/context cost 3')]:
        q=grid[grid.cost_scenario.eq(k)].sort_values('saving_vs_best_global_fixed')
        b.plot(np.arange(1,82),100*q.saving_vs_best_global_fixed,color=col,lw=1.2,label=l)
    b.set(xlim=(1,81),ylim=(0,8.4),xlabel='Rank within 81 stipulated intent priors',ylabel='Saving vs best global fixed order (%)');b.set_xticks([1,21,41,61,81]);clean(b);b.legend(loc='upper left',frameon=False,fontsize=6.7);label(b,'b','Sensitivity to costs and priors')
    b.text(0,-.42,'Against a per-family best fixed order: 0–0.096% saving.',transform=b.transAxes,fontsize=6.8)
    save(fig,'04_planning')


def coupling_audit():
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.05),gridspec_kw={'width_ratios':[1.05,1]});fig.subplots_adjust(left=.09,right=.977,bottom=.30,top=.85,wspace=.43)
    group=F.groupby('cohort');x=np.arange(12)
    raw=group.box_only_pairs.sum().reindex(COHORTS).values
    dependent=group.merged_box_only_pairs.sum().reindex(COHORTS).values
    equivalent=raw-dependent
    a=axs[0];a.barh(x,equivalent,color=GRAY,height=.69,label='Completion-equivalent functions');a.barh(x,dependent,left=equivalent,color=TEAL,height=.69,label='Dependence beyond equivalence')
    a.set(yticks=x,yticklabels=COHORTS,xlim=(0,410),ylim=(11.7,-.7),xlabel='Infeasible paired alarms (box only)');a.set_xticks([0,100,200,300,400]);clean(a);label(a,'a','Where the box loses dependence')
    for yi,n in enumerate(raw):a.text(n+5,yi,str(n),va='center',fontsize=6.4)
    a.legend(loc='upper left',bbox_to_anchor=(-.04,-.29),frameon=False,fontsize=6.1)
    b=axs[1];ov=group.observed_sensitive.sum().reindex(COHORTS).values;co=group.coupled_sensitive.sum().reindex(COHORTS).values
    b.barh(x,co,color=TEAL,height=.69,label='Completion-robust');b.barh(x,ov,color=INK,height=.31,label='Recorded-only')
    b.set(yticks=x,yticklabels=COHORTS,xlim=(0,32),ylim=(11.7,-.7),xlabel='Sensitive families (out of 30)');b.set_xticks([0,10,20,30]);clean(b);label(b,'b','Missingness changes the audit')
    b.legend(loc='upper left',bbox_to_anchor=(-.035,-.29),frameon=False,fontsize=6.1)
    save(fig,'05_coupling_audit')


def identification():
    fig,axs=plt.subplots(1,2,figsize=(7.15,2.5));fig.subplots_adjust(left=.075,right=.98,bottom=.35,top=.86,wspace=.47)
    a=axs[0];vals=[]
    for c in COHORTS:
        t=I[I.cohort.str.upper().eq(c)];vals.append(float(t.identification_width_pp.max()))
    a.bar(np.arange(12),vals,color=[RED if v>5 else TEAL for v in vals],width=.7)
    a.axhline(5,color=RED,lw=.8,ls='--');a.text(.02,.94,r'Point budget: $2\eta=5$ pp',transform=a.transAxes,fontsize=7,color=RED)
    a.set(xticks=range(12),xticklabels=COHORTS,ylim=(0,24),ylabel='Largest sharp identification width (pp)');a.tick_params(axis='x',rotation=55);clean(a);label(a,'a','Clarification does not collect assays')
    b=axs[1];v=I[I.cohort.eq('kirc')&I.gene.eq('VHL')].set_index('index')
    y=[2,1,0];inds=[0,4,5];names=['Profiled · broad','Cohort · broad','Cohort · restricted']
    for yi,ix,name in zip(y,inds,names):
        l,u=100*v.loc[ix,['lower','upper']].astype(float)
        b.plot([l,u],[yi,yi],color=TEAL if ix==0 else RED,lw=4,solid_capstyle='butt');b.scatter([l,u],[yi,yi],color=TEAL if ix==0 else RED,s=18,zorder=4)
        b.text(l,yi+.15,f'{l:.2f}%' if l==u else f'{l:.2f}–{u:.2f}%',fontsize=7.2,color=TEAL if ix==0 else RED)
    b.set(yticks=y,yticklabels=names,xlim=(15,58),ylim=(-.5,2.55),xlabel='VHL fraction in KIRC (%)');b.set_xticks([20,30,40,50]);clean(b);label(b,'b','Fixed estimands, different information')
    b.text(-.12,-.53,'720 / 5,760 selected specifications fail a 2.5 pp point-error budget.\nIntervals are identification sets, not confidence intervals.',transform=b.transAxes,fontsize=7.1,linespacing=1.5)
    save(fig,'06_identification')


def robustness():
    q=pd.read_csv(ROOT/'completion_results/threshold_summary.csv');grid=q.pivot(index='relative',columns='absolute_pp',values='optimal_expected_questions').sort_index()
    fig,axs=plt.subplots(1,2,figsize=(7.15,2.6));fig.subplots_adjust(left=.11,right=.92,bottom=.23,top=.84,wspace=.48)
    for ax,col,letter,title,vmin,vmax,fmt in [(axs[0],'sensitive_families','a','Sensitive families',300,360,'.0f'),(axs[1],'optimal_expected_questions','b','Expected questions',1,2.2,'.2f')]:
        a=q.pivot(index='relative',columns='absolute_pp',values=col).sort_index();im=ax.imshow(a.values,cmap='cividis',vmin=vmin,vmax=vmax,aspect='auto')
        for i in range(4):
            for j in range(4):ax.text(j,i,format(a.values[i,j],fmt),ha='center',va='center',fontsize=9,color='white' if (a.values[i,j]-vmin)/(vmax-vmin)<.5 else INK)
        ax.set(xticks=range(4),xticklabels=[1,2.5,5,10],yticks=range(4),yticklabels=['5%','10%','20%','40%'],xlabel='Absolute tolerance (pp)',ylabel='Relative tolerance');ax.tick_params(length=0)
        ax.add_patch(Rectangle((1.5,.5),1,1,fill=False,ec=RED,lw=1.8));label(ax,letter,title)
        fig.colorbar(im,ax=ax,shrink=.85,pad=.025).ax.tick_params(labelsize=7,length=2)
    fig.text(.11,.025,'Outlined cell: primary rule (10% relative OR 5 pp absolute). Every grid decision uses exact count support.',fontsize=7.1)
    save(fig,'07_tolerances')


def external():
    path=ROOT/'msk_results/families.csv'
    if not path.exists():return
    m=pd.read_csv(path);axis=pd.read_csv(ROOT/'msk_results/single_axis.csv')
    si=pd.read_csv(ROOT/'msk_results/interpretations.csv')
    clinical=pd.read_csv(ROOT/'data/msk/data_clinical_sample.txt',sep='\t',comment='#')
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.6));fig.subplots_adjust(left=.068,right=.979,bottom=.29,top=.85,wspace=.53)
    a=axs[0];pat=si[si['index'].eq(0)].set_index('gene');sam=si[si['index'].eq(8)].set_index('gene');x=100*pat.positive/pat.denominator;y=100*sam.positive/sam.denominator
    a.scatter(x,y,s=12,c=TEAL,alpha=.85,zorder=3);a.plot([0,60],[0,60],lw=.7,ls='--',c=GRAY)
    label_offsets={'TP53':((4,7),'left'),'KRAS':((-7,13),'right'),'PIK3CA':((8,-12),'left')}
    for gene,(offset,alignment) in label_offsets.items():
        a.annotate(gene,(x[gene],y[gene]),xytext=offset,textcoords='offset points',
                   fontsize=6.3,ha=alignment,va='center',
                   arrowprops={'arrowstyle':'-','lw':.45,'color':GRAY,'shrinkA':2,'shrinkB':2})
    a.set(xlim=(-1,58),ylim=(-1,58),xlabel='Patient fraction (%)',ylabel='Specimen fraction (%)');a.set_xticks([0,20,40]);a.set_yticks([0,20,40]);clean(a);label(a,'a','Repeated-specimen counting')
    a.text(.03,-.42,'Largest reference unit effect: 0.552 pp.',transform=a.transAxes,fontsize=6.6)
    b=axs[1];x=np.arange(4);cnt=axis.groupby('field').coupled_sensitive.sum().reindex(PROTOCOL['fields'])
    b.bar(x,cnt,color=[GRAY,GRAY,TEAL,INK],width=.66)
    for xi,n in enumerate(cnt):b.text(xi,n+1,str(n),ha='center',fontsize=7)
    b.set(xticks=x,xticklabels=['Unit','Assay','Context','Class'],ylim=(0,34),ylabel='Sensitive reference flips / 30');clean(b);label(b,'b','Retained scientific choices')
    b.text(.02,-.42,'All 30 genes are in both mutation panels.',transform=b.transAxes,fontsize=6.6)
    c=axs[2];mult=clinical.groupby('PATIENT_ID').size().value_counts().sort_index();c.bar(mult.index,mult.values,color=TEAL,width=.68)
    c.set_yscale('log');c.set(xlim=(.4,5.7),ylim=(.6,30000),xlabel='Specimens per patient',ylabel='Patient count (log scale)');c.set_xticks([1,2,3,4,5]);clean(c);label(c,'c','Clinical specimen multiplicity')
    for xi,n in zip(mult.index,mult.values):c.text(xi,n*1.35,f'{n:,}',ha='center',fontsize=6.4)
    c.text(.03,-.42,'562 repeated-specimen patients; maximum 5.',transform=c.transAxes,fontsize=6.6)
    save(fig,'08_targeted_assay')


def scaling():
    registry=pd.read_csv(ROOT/'scale_results/registry_scaling.csv')
    blocks=pd.read_csv(ROOT/'scale_results/block_scaling.csv')
    local=pd.read_csv(ROOT/'scale_results/local_block_scaling.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.28))
    fig.subplots_adjust(left=.075,right=.99,bottom=.27,top=.78,wspace=.57)
    a,b,c=axs
    for axis,color,marker,title in [('cohorts',TEAL,'o','Vary cohorts; 30 genes'),('genes',RED,'s','Vary genes; 12 cohorts')]:
        rows=registry.loc[registry.axis.eq(axis)].sort_values('families')
        y=rows.median_ms/1000
        a.errorbar(rows.families,y,yerr=np.array([y-rows.min_ms/1000,rows.max_ms/1000-y]),
                   color=color,marker=marker,ms=3.5,lw=1,capsize=2,label=title)
    a.set(xlabel='Cohort–gene families',ylabel='Audit + plan (s)',xlim=(0,382),ylim=(0,1.4))
    a.set_xticks([0,120,240,360]);a.legend(frameon=False,fontsize=6.9,loc='upper left',handlelength=1)
    clean(a);label(a,'a','Registry jobs')
    for implementation,color,marker in [('ungrouped',INK,'o'),('grouped',TEAL,'s')]:
        rows=blocks.loc[blocks.implementation.eq(implementation)].sort_values('blocks')
        b.plot(rows.blocks,rows.median_ms,color=color,marker=marker,ms=3.5,lw=1,label=implementation.title())
        b.fill_between(rows.blocks,rows.min_ms,rows.max_ms,color=color,alpha=.12)
    b.set(xscale='log',yscale='log',xlabel='Identical patient blocks',ylabel='One support call (ms)',ylim=(.004,4000))
    b.set_xticks([10,1000,100000]);b.set_xticklabels(['10',r'$10^3$',r'$10^5$'])
    b.legend(frameon=False,fontsize=7,loc='upper left');clean(b);label(b,'b','Multiplicity reuse')
    c.plot(local.unknown_specimens,local.median_ms,color=RED,marker='o',ms=3.5,lw=1)
    c.fill_between(local.unknown_specimens,local.min_ms,local.max_ms,color=RED,alpha=.12)
    c.set(yscale='log',xlabel='Unknown specimens / patient',ylabel='One support call (ms)',xlim=(.7,10.5),ylim=(.008,180))
    c.set_xticks([1,4,7,10]);clean(c);label(c,'c','Local block growth')
    c.text(.05,.94,r'$3^{10}=59{,}049$ local states',transform=c.transAxes,ha='left',va='top',fontsize=7)
    save(fig,'09_scaling')


if __name__=='__main__':
    for f in [architecture,matrix,geometry,planning,coupling_audit,identification,robustness,external,scaling]:f();print(f.__name__,flush=True)
