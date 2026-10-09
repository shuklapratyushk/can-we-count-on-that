"""An independent global LP check for the single-missing-specimen case.

This does not call the patient-block support formula. It constructs affine
query coefficients from specimens, expresses the global nested-class
constraints t <= b as a linear program, and checks every directed support
needed by the primary absolute and relative decisions. Zero/identical
objectives are cached; all family-pair checks remain in the audit count.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from engine import PROTOCOL,ASSIGNMENTS,Spec


def global_affine_coefficients(samples,mutations,cohort,gene):
    rows=samples.loc[samples.cohort.eq(cohort)]
    missing=rows.loc[~rows.profiled].copy()
    if len(missing) and missing.groupby('patient').size().max()>1:
        raise ValueError("Affine LP reference requires at most one unprofiled specimen per patient")
    m=len(missing);coeff=np.zeros((16,2*m),dtype=np.int64)
    base=np.zeros(16,dtype=np.int64);denom=np.zeros(16,dtype=np.int64)
    gene_records=mutations.loc[mutations.cohort.eq(cohort)&mutations.gene.eq(gene)]
    for i,bits in enumerate(ASSIGNMENTS):
        spec=Spec.from_bits(cohort,gene,bits)
        context=rows if spec.sample_context=='all' else rows.loc[rows.primary]
        measured=context.loc[context.profiled]
        eligible=measured if spec.assay_denominator=='profiled' else context
        denom[i]=eligible[spec.analysis_unit].nunique()
        records=gene_records.loc[gene_records.mutation_type.isin(PROTOCOL[spec.mutation_scope+'_types'])]
        positive_samples=set(records['sample'])&set(measured['sample'])
        positive_units=set(measured.loc[measured['sample'].isin(positive_samples),spec.analysis_unit])
        base[i]=len(positive_units)
        if spec.assay_denominator=='cohort':
            for j,(_,row) in enumerate(missing.iterrows()):
                if spec.sample_context=='primary' and not row['primary']:continue
                if spec.analysis_unit=='patient' and row['patient'] in positive_units:continue
                coeff[i,2*j+(spec.mutation_scope=='truncating')]=1
    return base,denom,coeff


@lru_cache(maxsize=100000)
def _global_lp_support(key):
    pairs=np.asarray(key,dtype=np.int64).reshape(-1,2)
    m=len(pairs)
    if not m:return 0
    objective=pairs.ravel().astype(float)
    # Each row is t_j - b_j <= 0; all b_j,t_j lie in [0,1].
    rr=np.repeat(np.arange(m),2);cc=np.arange(2*m);vv=np.tile([-1.,1.],m)
    A=coo_matrix((vv,(rr,cc)),shape=(m,2*m)).tocsr()
    answer=linprog(-objective,A_ub=A,b_ub=np.zeros(m),bounds=(0,1),method='highs')
    if not answer.success:raise RuntimeError(answer.message)
    support=float(objective@answer.x)
    integer=int(round(support))
    if abs(support-integer)>1e-5:raise AssertionError("Nested-class LP optimum was not integral")
    return integer


def lp_support(coefficients):
    pairs=np.asarray(coefficients,dtype=np.int64).reshape(-1,2)
    # Reordering independent constraints and removing zero objectives changes
    # neither the global LP feasible set's objective optimum nor its value.
    nonzero=sorted(tuple(map(int,row)) for row in pairs if np.any(row))
    return _global_lp_support(tuple(value for row in nonzero for value in row))


def verify_model(model,samples,mutations):
    base,n,A=global_affine_coefficients(samples,mutations,model.cohort,model.gene)
    np.testing.assert_array_equal(n,model.denominators)
    np.testing.assert_array_equal(base,model.observed_counts)
    supports=[(1,-1,model.support_numerator(1,-1)),
              (19,-21,model.support_numerator(19,-21))]
    count=0
    for i in range(16):
        for j in range(i+1,16):
            for a,b,expected in supports:
                for x,y in [(i,j),(j,i)]:
                    value=(a*int(base[x])*int(n[y])+b*int(base[y])*int(n[x])
                           +lp_support(a*A[x]*n[y]+b*A[y]*n[x]))
                    assert value==expected[x,y],(model.cohort,model.gene,x,y,a,b,value,expected[x,y])
            count+=1
    return count


def cache_stats():
    info=_global_lp_support.cache_info()
    return {'distinct_global_lp_objectives':info.misses,'reused_global_lp_objectives':info.hits,
            'method':'Global HiGHS LP with nested-class constraints; exact integer support comparisons'}
