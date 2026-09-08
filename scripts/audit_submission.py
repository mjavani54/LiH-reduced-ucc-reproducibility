#!/usr/bin/env python3
"""Audit deposited records and spin-partner closure; does not rerun VQE."""
from pathlib import Path
import argparse,csv,json

def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def spin_partner(i):
    if i<9:return i+9
    if i<18:return i-9
    a,b=divmod(i-18,9)
    return 18+9*b+a

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();r=args.root
    frozen=json.loads((r/'results/phase7c/phase7c_frozen_discovery_support.json').read_text())
    compact=frozen['transferred_compact10_global_indices']
    prefix=frozen['external_ranking_global_indices'][:37]
    candidate=set(compact+prefix);boundary=candidate-{17}
    assert len(candidate)==47
    assert all(spin_partner(i) in candidate for i in candidate)
    assert {spin_partner(i) for i in boundary}-boundary=={17}
    controls=read_csv(r/'results/phase7c/phase7c_random_support_summary.csv')
    bank={x['path_id']:x for x in frozen['random_control']['paths']}
    n_single=sum(i<18 for i in prefix);n_double=37-n_single
    closures=[]
    for row in controls:
        name=row['support_name']
        if not name.endswith('_composition'):
            raise ValueError('A fallback control requires separate reconstruction: '+name)
        path=bank[name.removesuffix('_composition')]
        ids=set(compact+path['external_single_order'][:n_single]+path['external_double_order'][:n_double])
        assert len(ids)==47 and sum(i<18 for i in ids)==10
        closures.append(all(spin_partner(i) in ids for i in ids))
    summaries=read_csv(r/'results/phase7d/phase7d_variant_geometry_summary.csv')
    assert len(summaries)==12
    for row in summaries:
        delta=float(row['best_variational_total_ha'])-float(row['accepted_full_2e10o_exact_total_ha'])
        assert abs(delta-float(row['best_error_vs_full_exact_ha']))<1e-12
        if row['variant']=='spin_complete_candidate47':
            assert abs(delta)<=1e-6 and float(row['best_spin_square'])<=1e-7
    runs=read_csv(r/'results/phase7d/phase7d_restart_runs.csv')
    assert len(runs)==24
    result={'scope':'Deposited records only; no electronic-structure or VQE rerun',
      'confirmation_rows':len(summaries),'restart_rows':len(runs),
      'candidate47_partner_closed':True,'boundary46_missing_partners':[17],
      'random_controls':len(controls),'partner_closed_random_controls':sum(closures),
      'max_final_gradient_infinity_norm':max(float(x['gradient_infinity_norm']) for x in runs),
      'optimizer_messages':sorted(set(x['optimizer_message'] for x in runs))}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
