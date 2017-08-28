import json
import time
import sys
import os
TIMESTAMP = str(int(time.time()))


def read_json(filepath1, filepath2, outdir):
    with open(filepath1) as fp1, open(filepath2) as fp2:
        jds1 = json.load(fp1)
        jds2 = json.load(fp2)
        pp = []
        nn = []
        pn = []
        np = []
        for jd1, jd2 in zip(jds1, jds2):
            assert jd1['true_q'] == jd2['true_q']
            if jd1['match'] == jd2['match'] == 1:
                pp.append({'true_q': jd1['true_q'],
                           'true_prob_1':jd1['true_prob'],
                           'true_prob_2':jd2['true_prob']
                })
            elif jd1['match'] == jd2['match'] == 0:
                nn.append({'true_q': jd1['true_q'],
                           'true_prob_1':jd1['true_prob'],
                           'true_prob_2':jd2['true_prob'],
                           'max_q_1': jd1['max_q'],
                           'max_q_2': jd2['max_q'],
                           'max_prob_1':jd1['max_prob'],
                           'max_prob_2':jd2['max_prob']})
            elif jd1['match'] == 1 and jd2['match'] == 0:
                pn.append({'true_q': jd1['true_q'],
                           'true_prob_1':jd1['true_prob'],
                           'true_prob_2':jd2['true_prob'],
                           'max_q_2': jd2['max_q'],
                           'max_prob_2':jd2['max_prob']})
            elif jd1['match'] == 0 and jd2['match'] == 1:
                np.append({'true_q': jd1['true_q'],
                           'true_prob_1':jd1['true_prob'],
                           'true_prob_2':jd2['true_prob'],
                           'max_q_1': jd1['max_q'],
                           'max_prob_1':jd1['max_prob']})
    with open(os.path.join(outdir, 'pp_{}.json'.format(TIMESTAMP)), 'w') as fp:
        json.dump(pp, fp)
    with open(os.path.join(outdir, 'nn_{}.json'.format(TIMESTAMP)), 'w') as fp:
        json.dump(nn, fp)
    with open(os.path.join(outdir, 'pn_{}.json'.format(TIMESTAMP)), 'w') as fp:
        json.dump(pn, fp)
    with open(os.path.join(outdir, 'np_{}.json'.format(TIMESTAMP)), 'w') as fp:
        json.dump(np, fp)

if __name__ == '__main__':
    fpath1 = sys.argv[1]
    fpath2 = sys.argv[2]
    outdir = sys.argv[3]
    read_json(fpath1, fpath2, outdir)
