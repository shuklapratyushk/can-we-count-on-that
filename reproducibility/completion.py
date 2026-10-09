"""Coupled missing-status certificates for typed cancer-genomics queries.

Missing specimen statuses are shared by all interpretations. For each gene,
an unprofiled specimen has one of three nested event states: absent,
broad-only, or restricted-class (and therefore broad; legacy code token truncating). Patient-level positivity is
an OR over the eligible specimens. Global uncertainty factors by patient.

No statistical or clinical prevalence prior is imposed on these completions.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product, permutations
import numpy as np

from engine import PROTOCOL, ASSIGNMENTS, FIELDS, Spec, PandasEngine


class CompletionModel:
    def __init__(self, samples, mutations, cohort, gene, baseline_results=None,
                 max_block_states=100000):
        self.cohort, self.gene = cohort, gene
        self.samples = samples.loc[samples.cohort.eq(cohort)].copy()
        self.mutations = mutations.loc[mutations.cohort.eq(cohort) & mutations.gene.eq(gene)].copy()
        self.specs = [Spec.from_bits(cohort, gene, b) for b in ASSIGNMENTS]
        if baseline_results is None:
            engine = PandasEngine(self.samples, self.mutations)
            baseline_results = [engine.execute(s) for s in self.specs]
        self.denominators = np.array([r.denominator for r in baseline_results],dtype=np.int64)
        self.observed_counts = np.array([r.positive for r in baseline_results],dtype=np.int64)
        if (self.denominators <= 0).any():
            raise ValueError("Undefined specification: a zero denominator cannot receive a certificate")
        known = {}
        eligible_samples = set(self.samples.loc[self.samples.profiled, "sample"])
        for scope in ("broad", "truncating"):
            rows = self.mutations.loc[self.mutations.mutation_type.isin(PROTOCOL[scope+"_types"])]
            known[scope] = set(rows['sample']) & eligible_samples
        self.blocks=[]
        for patient,group in self.samples.groupby("patient",sort=True):
            unknown=group.loc[~group.profiled,"sample"].tolist()
            if not unknown:
                continue
            if 3**len(unknown)>max_block_states:
                raise ValueError("Patient completion block exceeds the declared enumeration budget")
            assignments=list(product((0,1,2),repeat=len(unknown)))
            counts=np.zeros((len(assignments),16),dtype=np.int64)
            contexts={}
            for context in ("all","primary"):
                rows=group if context=="all" else group.loc[group.primary]
                for assay in ("profiled","cohort"):
                    use=rows.loc[rows.profiled] if assay=="profiled" else rows
                    contexts[(context,assay)]=set(use['sample'])
            for j,state in enumerate(assignments):
                broad=known['broad'] | {s for s,z in zip(unknown,state) if z>=1}
                trunc=known['truncating'] | {s for s,z in zip(unknown,state) if z==2}
                for i,spec in enumerate(self.specs):
                    positives=contexts[(spec.sample_context,spec.assay_denominator)] & (broad if spec.mutation_scope=='broad' else trunc)
                    counts[j,i] = int(bool(positives)) if spec.analysis_unit=='patient' else len(positives)
            self.blocks.append({'patient':patient,'unknown_samples':unknown,
                                'states':assignments,'counts':counts})
        self.fixed_counts = self.observed_counts.copy()
        for block in self.blocks:
            self.fixed_counts -= block['counts'][0]
        assert (self.fixed_counts>=0).all()
        self.lower_counts=self.fixed_counts.copy()
        self.upper_counts=self.fixed_counts.copy()
        for block in self.blocks:
            self.lower_counts += block['counts'].min(axis=0)
            self.upper_counts += block['counts'].max(axis=0)
        assert np.array_equal(self.lower_counts,self.observed_counts)
        self.observed = self.observed_counts/self.denominators
        self.lower=self.lower_counts/self.denominators
        self.upper=self.upper_counts/self.denominators
        self.max_unprofiled_per_patient=max([len(b['unknown_samples']) for b in self.blocks],default=0)
        self.local_state_count=sum(len(b['states']) for b in self.blocks)
        self.unprofiled_samples=sum(len(b['unknown_samples']) for b in self.blocks)
        self._grouped=[]
        # Group identical contribution tables without losing the patient-block
        # interpretation. The multiplicity is a computational compression only.
        groups={}
        for block in self.blocks:
            k=block['counts'];key=(k.shape,k.tobytes())
            if key not in groups:groups[key]=[k,0]
            groups[key][1]+=1
        self._grouped=list(groups.values())

    def support_numerator(self, first_weight=1, second_weight=-1):
        """All directed pair supports, scaled by n_i*n_j, using integer counts."""
        n=self.denominators
        value=first_weight*self.fixed_counts[:,None]*n[None,:]+second_weight*self.fixed_counts[None,:]*n[:,None]
        for counts,multiplicity in self._grouped:
            terms=(first_weight*counts[:,:,None]*n[None,None,:]
                   +second_weight*counts[:,None,:]*n[None,:,None])
            value += multiplicity*terms.max(axis=0)
        return value

    def absolute_certificate(self):
        directed=self.support_numerator(1,-1)
        numerator=np.maximum(directed,directed.T)
        return 100*numerator/(self.denominators[:,None]*self.denominators[None,:]),numerator

    def relative_threshold(self, threshold=0.10, epsilon=1e-12):
        """Strict threshold decisions with integer support and exact rational epsilon."""
        tau=Fraction(str(threshold));a,b=tau.numerator,tau.denominator
        directed=self.support_numerator(2*b-a,-(2*b+a))
        numerator=np.maximum(directed,directed.T)
        # Exact comparison after scaling; no floating-point root is used to
        # decide sensitivity. The scientific counts fit signed int64 here.
        eps=Fraction(str(epsilon))
        products=self.denominators[:,None]*self.denominators[None,:]
        flags=np.empty((16,16),dtype=bool)
        for i in range(16):
            for j in range(16):
                flags[i,j]=Fraction(int(numerator[i,j]),1)>2*a*eps*int(products[i,j])
        np.fill_diagonal(flags,False)
        return flags,numerator

    def absolute_threshold(self, threshold_pp=5.0):
        _,numerator=self.absolute_certificate()
        tau=Fraction(str(threshold_pp))/100
        products=self.denominators[:,None]*self.denominators[None,:]
        return numerator*tau.denominator > tau.numerator*products

    def relative_certificate(self, iterations=40, epsilon=1e-12):
        """Bracket all directed fractional optima; positive-part directions suffice."""
        n=self.denominators.astype(float)
        fixed=self.fixed_counts/n
        constant_diff=fixed[:,None]-fixed[None,:]
        constant_mean=(fixed[:,None]+fixed[None,:])/2
        group_terms=[]
        for counts,multiplicity in self._grouped:
            values=counts/n
            group_terms.append((values[:,:,None]-values[:,None,:],
                                (values[:,:,None]+values[:,None,:])/2,multiplicity))
        def phi(rho):
            value=constant_diff-rho*constant_mean-rho*epsilon
            for delta,mean,multiplicity in group_terms:
                value+=multiplicity*np.max(delta-rho[None,:,:]*mean,axis=0)
            return value
        lo=np.zeros((16,16));hi=np.where(phi(lo)>0,2.,0.)
        for _ in range(iterations):
            mid=(lo+hi)/2
            above=phi(mid)>0
            lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
        lower=np.maximum(lo,lo.T);upper=np.maximum(hi,hi.T)
        np.fill_diagonal(lower,0);np.fill_diagonal(upper,0)
        return lower,upper

    def certificate_graph(self,relative=0.10,absolute_pp=5.0):
        flags,_=self.relative_threshold(relative)
        flags |= self.absolute_threshold(absolute_pp)
        np.fill_diagonal(flags,False)
        return flags

    def observed_graph(self,relative=0.10,absolute_pp=5.0):
        # Known counts are an independent, degenerate completion model.
        n=self.denominators;k=self.observed_counts
        products=n[:,None]*n[None,:]
        absn=np.abs(k[:,None]*n[None,:]-k[None,:]*n[:,None])
        tau=Fraction(str(absolute_pp))/100
        absflags=absn*tau.denominator>tau.numerator*products
        tau=Fraction(str(relative));a,b=tau.numerator,tau.denominator
        dirs=(2*b-a)*k[:,None]*n[None,:]-(2*b+a)*k[None,:]*n[:,None]
        reln=np.maximum(dirs,dirs.T)
        eps=Fraction('0.000000000001')
        flags=absflags.copy()
        for i in range(16):
            for j in range(16):
                flags[i,j] |= Fraction(int(reln[i,j]),1)>2*a*eps*int(products[i,j])
        np.fill_diagonal(flags,False)
        return flags

    def independent_interval_certificate(self):
        absolute=100*np.maximum(self.upper[:,None]-self.lower[None,:],
                                self.upper[None,:]-self.lower[:,None])
        def ratio(a,b):return np.maximum(a-b,0)/((a+b)/2+1e-12)
        relative=np.maximum(ratio(self.upper[:,None],self.lower[None,:]),
                            ratio(self.upper[None,:],self.lower[:,None]))
        np.fill_diagonal(absolute,0);np.fill_diagonal(relative,0)
        return absolute,relative

    def independent_interval_graph(self,merge_equivalent=False,relative=0.10,absolute_pp=5.0):
        n=self.denominators;u=self.upper_counts;l=self.lower_counts
        products=n[:,None]*n[None,:]
        absolute=np.maximum(u[:,None]*n[None,:]-l[None,:]*n[:,None],
                            u[None,:]*n[:,None]-l[:,None]*n[None,:])
        tau=Fraction(str(absolute_pp))/100
        graph=absolute*tau.denominator>tau.numerator*products
        tau=Fraction(str(relative));a,b=tau.numerator,tau.denominator
        directed=(2*b-a)*u[:,None]*n[None,:]-(2*b+a)*l[None,:]*n[:,None]
        relative_numerator=np.maximum(directed,directed.T)
        eps=Fraction('0.000000000001')
        for i in range(16):
            for j in range(16):
                graph[i,j] |= Fraction(int(relative_numerator[i,j]),1)>2*a*eps*int(products[i,j])
        if merge_equivalent:
            # Give this outer-bound comparator exact equivalence information.
            # This favors the baseline and isolates nontrivial shared-status
            # dependence beyond the obvious identical-output cases.
            _,numerator=self.absolute_certificate()
            graph &= numerator!=0
        np.fill_diagonal(graph,False)
        return graph

    def completion_values(self,state_indices):
        counts=self.fixed_counts.copy()
        for block,index in zip(self.blocks,state_indices):counts+=block['counts'][index]
        return counts/self.denominators

    def support_witness(self,i,j,relative_rho=None):
        if relative_rho is None:
            coefficients=np.array([1.,-1.])
        else:
            coefficients=np.array([1-relative_rho/2,-1-relative_rho/2])
        indices=[]
        for block in self.blocks:
            values=block['counts'][:,[i,j]]/self.denominators[[i,j]]
            indices.append(int(np.argmax(values@coefficients)))
        values=self.completion_values(indices)
        return values,indices


class GraphPlanner:
    """Exact stopping/clarification planning on the violating-pair graph."""
    def __init__(self,graph,weights=None,costs=(1,1,1,1)):
        self.graph=np.asarray(graph,dtype=bool)
        assert self.graph.shape==(16,16)
        assert np.array_equal(self.graph,self.graph.T) and not np.diag(self.graph).any()
        self.weights=np.asarray(weights if weights is not None else np.ones(16)/16,dtype=float)
        if (self.weights<=0).any():raise ValueError("All supported intents need positive weight")
        self.weights=self.weights/self.weights.sum()
        self.costs=tuple(float(c) for c in costs)
        if len(self.costs)!=4 or min(self.costs)<=0:raise ValueError("Positive cost required for each field")
        self.root=tuple(range(16))
        self._solutions={}
        self._partitions={}

    def stable(self,state):
        return not self.graph[np.ix_(state,state)].any()

    def partition(self,state,field):
        key=(state,field)
        if key not in self._partitions:
            self._partitions[key]=tuple(tuple(i for i in state if ASSIGNMENTS[i][field]==value) for value in (0,1))
        return self._partitions[key]

    def admissible(self,state):
        return [f for f in range(4) if all(self.partition(state,f))]

    def solve(self,state,mode='expected'):
        key=(state,mode)
        if key in self._solutions:return self._solutions[key]
        if self.stable(state):return 0.,None
        candidates=[]
        mass=self.weights[list(state)].sum()
        for field in self.admissible(state):
            groups=self.partition(state,field)
            values=[self.solve(group,mode)[0] for group in groups]
            if mode=='expected':
                future=sum(self.weights[list(group)].sum()/mass*value for group,value in zip(groups,values))
            elif mode=='minimax':future=max(values)
            else:raise ValueError("Unknown planning mode")
            candidates.append((self.costs[field]+future,field))
        answer=min(candidates)
        self._solutions[key]=answer
        return answer

    def edge_mass(self,state):
        w=self.weights[list(state)];w=w/w.sum()
        g=self.graph[np.ix_(state,state)]
        return float((w[:,None]*w[None,:]*g).sum()/2)

    def heuristic(self,state,mode):
        mass=self.weights[list(state)].sum()
        choices=[]
        for field in self.admissible(state):
            groups=self.partition(state,field)
            probs=[self.weights[list(group)].sum()/mass for group in groups]
            if mode=='edge_greedy':
                # One-step reduction in the probability that two independent
                # conditional intent draws form a violating pair.
                after=sum(p*self.edge_mass(group) for p,group in zip(probs,groups))
                score=(self.edge_mass(state)-after)/self.costs[field]
            elif mode=='entropy':score=-sum(p*np.log2(p) for p in probs)/self.costs[field]
            else:raise ValueError("Unknown heuristic")
            choices.append((-score,field))
        return min(choices)[1]

    def walk(self,truth,policy='expected',order=None):
        state=self.root;fields=[];total=0.
        while not self.stable(state):
            if order is not None:
                field=next(f for f in order if f in self.admissible(state))
            elif policy in ('expected','minimax'):field=self.solve(state,policy)[1]
            else:field=self.heuristic(state,policy)
            fields.append(field);total+=self.costs[field]
            state=tuple(i for i in state if ASSIGNMENTS[i][field]==ASSIGNMENTS[truth][field])
        assert truth in state and self.stable(state)
        return total,fields,state

    def policy_costs(self,policy='expected',order=None):
        values=np.array([self.walk(i,policy,order)[0] for i in range(16)])
        return float(self.weights@values),float(values.max()),values


def product_prior(zero_probabilities):
    return np.array([np.prod([p if bit==0 else 1-p for p,bit in zip(zero_probabilities,bits)])
                     for bits in ASSIGNMENTS],dtype=float)


# All candidate subsets obtainable by truthful binary field replies.
STATES=sorted({tuple(i for i,bits in enumerate(ASSIGNMENTS)
                     if all(value==-1 or bits[f]==value for f,value in enumerate(partial)))
               for partial in product((-1,0,1),repeat=4)},key=lambda s:(len(s),s))
STATE_INDEX={state:i for i,state in enumerate(STATES)}
MEMBERSHIP=np.array([[int(i in state) for state in STATES] for i in range(16)])
STATE_ACTIONS=[]
for state in STATES:
    actions=[]
    for field in range(4):
        children=tuple(tuple(i for i in state if ASSIGNMENTS[i][field]==value) for value in (0,1))
        if all(children):actions.append((field,*(STATE_INDEX[c] for c in children)))
    STATE_ACTIONS.append(actions)


def expected_cost_grid(graph,weights,costs):
    """Vectorized exact Bellman recurrence for many positive intent priors."""
    weights=np.asarray(weights,dtype=float)
    if weights.ndim==1:weights=weights[None,:]
    if (weights<=0).any():raise ValueError("Positive intent weights required")
    weights=weights/weights.sum(axis=1,keepdims=True)
    masses=weights@MEMBERSHIP
    values=np.zeros((len(weights),len(STATES)))
    choices=np.full((len(weights),len(STATES)),-1,dtype=np.int64)
    for index,state in enumerate(STATES):
        if not graph[np.ix_(state,state)].any():continue
        candidates=[];fields=[]
        for field,zero,one in STATE_ACTIONS[index]:
            candidate=(costs[field]+masses[:,zero]/masses[:,index]*values[:,zero]
                       +masses[:,one]/masses[:,index]*values[:,one])
            candidates.append(candidate);fields.append(field)
        table=np.stack(candidates,axis=1)
        argmin=table.argmin(axis=1)
        values[:,index]=table[np.arange(len(weights)),argmin]
        choices[:,index]=np.asarray(fields)[argmin]
    root=STATE_INDEX[tuple(range(16))]
    return values[:,root],choices[:,root]
