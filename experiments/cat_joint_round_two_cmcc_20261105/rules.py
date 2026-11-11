"""Confirmed protocol section4 arithmetic; no synthetic model measurements."""
from fractions import Fraction
import itertools
import math

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_offline_execution_cmcc_20261104.selection import quality_key

RATES=(Fraction(2),Fraction(5,2),Fraction(3))


def stable(row):
    return tuple((category,Fraction(rate)) for category,rate in row['rate_vector'])


def enumerate_pool(inventory,profiles,baseline):
    categories=sorted(inventory['category_ids']);options={category:{} for category in categories}
    sizes={category:sum(item['numel'] for item in inventory['projections'] if item['category_id']==category) for category in categories}
    require(len(profiles)==len(categories)*3,'Require all announced three-rate profiles, including negative observations.')
    for row in profiles:
        category=row['category_id'];rate=Fraction(row['rate']);fee=row['actual_class_fee']
        require(row['status']=='complete' and category in options and rate in RATES and rate not in options[category]
            and fee['N']==sizes[category] and row['bank_digest'],'Invalid/incomplete/duplicate actual category-rate source.')
        options[category][rate]=row
    require(all(set(rows)==set(RATES) for rows in options.values()),'Missing actual rate coverage.')
    n=inventory['total_target_numel'];working=math.floor(Fraction(5,2)*n);hard=math.floor(Fraction(13,5)*n)
    candidates=[]
    for values in itertools.product(*(options[category].values() for category in categories)):
        rates=[[category,str(Fraction(row['rate']))] for category,row in zip(categories,values)]
        banks={category:row['bank_digest'] for category,row in zip(categories,values)}
        logical=sum(row['actual_class_fee']['logical_bits'] for row in values)
        if logical>hard:continue
        signed=sum(Fraction(str(baseline))-Fraction(str(row['full8']['mean_percent'])) for row in values)
        candidates.append(dict(proposal_id=digest(dict(rates=rates,banks=banks)),rate_vector=rates,logical_bits=logical,
            packed_bits=sum(row['actual_class_fee']['packed_bits'] for row in values),signed_profile_loss=signed,profiles=dict(zip(categories,values))))
    require(candidates,'No actual rate vector satisfies the original logical2.6 hard limit.')
    preferred=[row for row in candidates if row['logical_bits']<=working]
    pool=preferred or candidates;limit=working if preferred else hard
    identity=digest(dict(limit=limit,vectors=sorted(row['proposal_id'] for row in pool),N=n))
    return pool,dict(N=n,working_limit_bits=working,hard_limit_bits=hard,common_limit_bits=limit,working_target_feasible=bool(preferred),
        hard_feasible_count=len(candidates),common_set_count=len(pool),common_set_identity=identity)


def nominations(pool,parent,measured):
    lookup={stable(row):row for row in pool};base=dict(stable(parent));categories=sorted(base);chosen={}
    require(stable(parent) in lookup,'Actual R1 parent is outside the original common working set.')
    used={stable(row) for row in measured}
    for up,down in itertools.permutations(categories,2):
        rates=dict(base);rates[up]+=Fraction(1,2);rates[down]-=Fraction(1,2)
        if rates[up] not in RATES or rates[down] not in RATES:continue
        vector=tuple(sorted(rates.items()))
        if vector in used or vector not in lookup:continue
        row=lookup[vector];chosen[row['proposal_id']]=row
    return sorted(chosen.values(),key=lambda row:(row['signed_profile_loss'],row['logical_bits'],row['packed_bits'],stable(row)))[:4]


def rank_actual(rows):
    require(rows and len({row['proposal_id'] for row in rows})==len(rows),'Rank the complete distinct measured set.')
    return sorted(rows,key=lambda row:quality_key(row['full8_mean'],row['logical_bits'],row['packed_bits'],stable(row)))


def allocate_Nc(inventory,ledger):
    categories=sorted(inventory['category_ids']);n=inventory['total_target_numel'];hard=math.floor(Fraction(13,5)*n)
    require(ledger['N']==n and set(ledger['per_category'])==set(categories),'Use actual full-inventory winner ledger.')
    fees=ledger['per_category'];sizes={category:sum(row['numel'] for row in inventory['projections'] if row['category_id']==category) for category in categories}
    require(all(fees[category]['N']==sizes[category] for category in categories) and sum(sizes.values())==n
        and sum(row['logical_bits'] for row in fees.values())==ledger['logical_bits']<=hard,'Actual category fees/denominator do not close.')
    reserve=hard-ledger['logical_bits'];shares={category:reserve*sizes[category]//n for category in categories}
    remainder=reserve-sum(shares.values());require(0<=remainder<len(categories),'Unexpected integer floor allocation remainder.')
    for category in categories[:remainder]:shares[category]+=1
    caps={category:fees[category]['logical_bits']+shares[category] for category in categories}
    require(sum(caps.values())==hard,'Nc caps must exhaust exactly the original hard budget.')
    return caps,dict(method='floor(R*Nc/N), remaining bits one each in stable category-ID order',R=reserve,N=n,H=hard,
        floor_remainder_bits=remainder,stable_category_order=categories,reserve_by_category=shares)
