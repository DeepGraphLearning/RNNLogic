import copy
import gc
from collections import defaultdict

import torch

import groundings
from knowledge_graph_utils import mask2list, list2mask, build_graph
from metrics import Metrics
from reasoning_model import ReasoningModel
from rotate import RotatE


class RNNLogic(torch.nn.Module):
    def __init__(self, dataset, args, print=print):
        super(RNNLogic, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R']
        assert self.R % 2 == 0
        args = self.set_args(args)

        self.dataset = dataset
        self._print = print
        self.print = self.log_print
        self.predictor_init = lambda: RNNLogicPredictor(self.dataset, self._args, print=self.log_print)
        self.generator = RNNLogicGenerator(self.R, self.arg('generator_embed_dim'), self.arg('generator_hidden_dim'),
                                           print=self.log_print)

        print = self.print
        print("RNNLogic Init", self.E, self.R)

    def log_print(self, *args, **kwargs):
        import datetime
        timestr = datetime.datetime.now().strftime("%H:%M:%S.%f")
        if hasattr(self, 'em'):
            emstr = self.em if self.em < self.num_em_epoch else '#'
            prefix = f"r = {self.r} EM = {emstr}"
        else:
            prefix = "init"
        self._print(f"[{timestr}] {prefix} | ", end="")
        self._print(*args, **kwargs)

    # Use EM algorithm to train RNNLogic model
    def train_model(self, r, num_em_epoch=None, rule_file=None, model_file=None):
        if rule_file is None:
            rule_file = f"rules_{r}.txt"
        if model_file is None:
            model_file = f"model_{r}.pth"
        if num_em_epoch is None:
            num_em_epoch = self.arg('num_em_epoch')

        self.num_em_epoch = num_em_epoch
        self.r = r
        print = self.print

        pgnd_buffer = dict()
        rgnd_buffer = dict()
        rgnd_buffer_test = dict()

        def generate_rules():
            if self.em == 0:
                self.predictor.relation_init(r=r, rule_file=rule_file, force_init_weight=self.arg('init_weight_boot'))
            else:
                sampled = set()
                sampled.add((r,))
                sampled.add(tuple())

                rules = [(r,)]
                prior = [0.0, ]
                for rule, score in self.generator.beam_search(r,
                                                              self.arg('max_beam_rules'),
                                                              self.predictor.arg('max_rule_len')):
                    rule = tuple(rule)
                    if rule in sampled:
                        continue
                    sampled.add(rule)
                    rules.append(rule)
                    prior.append(score)
                    if len(sampled) % self.arg('sample_print_epoch') == 0:
                        print(f"sampled # = {len(sampled)} rule = {rule} score = {score}")

                print(f"Done |sampled| = {len(sampled)}")

                prior = torch.tensor(prior).cuda()
                prior -= prior.max()
                prior = prior.exp()

                self.predictor.relation_init(r, rules=rules, prior=prior)


        for self.em in range(num_em_epoch):
            self.predictor = self.predictor_init()
            self.predictor.pgnd_buffer = pgnd_buffer
            self.predictor.rgnd_buffer = rgnd_buffer
            self.predictor.rgnd_buffer_test = rgnd_buffer_test

            
            generate_rules()

            # E-Step:
            valid, test = self.predictor.train_model()

            # M-Step
            gen_batch = self.predictor.make_gen_batch()
            self.generator.train_model(gen_batch,
                                       lr=self.arg('generator_lr'),
                                       num_epoch=self.arg('generator_num_epoch'),
                                       print_epoch=self.arg('generator_print_epoch'))

            ckpt = {
                'r': r,
                'metrics': {
                    'valid': valid,
                    'test': test
                },
                'args': self._args_init,
                'rules': self.predictor.rules_exp,
                'predictor': self.state_dict(),
            }
            torch.save(ckpt, model_file)
            gc.collect()

        # Testing
        self.em = num_em_epoch
        generate_rules()
        valid, test = self.predictor.train_model()

        gc.collect()
        return valid, test

    def arg(self, name, apply=None):
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    # Definitions for EM framework
    def set_args(self, args):
        self._args_init = args
        self._args = dict()
        def_args = dict()
        def_args['num_em_epoch'] = 3
        def_args['sample_print_epoch'] = 100
        def_args['max_beam_rules'] = 3000
        def_args['generator_embed_dim'] = 512
        def_args['generator_hidden_dim'] = 256
        def_args['generator_lr'] = 1e-3
        def_args['generator_num_epoch'] = 10000
        def_args['generator_print_epoch'] = 100
        def_args['init_weight_boot'] = False

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in def_args.items():
            self._args[k] = str(v) if make_str(v) else v
        for k, v in args.items():
            # if k not in self._args:
            # 	print(f"Warning: Unused argument '{k}'")
            self._args[k] = str(v) if make_str(v) else v


class RNNLogicPredictor(ReasoningModel):
    def __init__(self, dataset, args, print=print):
        super(RNNLogicPredictor, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R'] + 1
        assert self.R % 2 == 1
        self.dataset = dataset

        self.set_args(args)
        rotate_pretrained = self.arg('rotate_pretrained', apply=lambda x: x)
        self.rotate = RotatE(dataset, rotate_pretrained)
        self.training = True

        self.rule_weight_raw = torch.nn.Parameter(torch.zeros(1))
        if rotate_pretrained is not None:
            if self.arg('param_relation_embed'):
                self.rotate.enable_parameter('relation_embed')
            if self.arg('param_entity_embed'):
                self.rotate.enable_parameter('entity_embed')

        self.pgnd_buffer = dict()
        self.rgnd_buffer = dict()
        self.rgnd_buffer_test = dict()
        self.cuda()
        self.print = print
        self.debug = False
        self.recording = False

    def train(self, mode=True):
        self.training = mode
        super(RNNLogicPredictor, self).train(mode)

    def eval(self):
        self.train(False)

    def index_select(self, tensor, index):
        if self.training:
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index)
            index = index.to(tensor.device)
            return tensor.index_select(0, index).squeeze(0)
        else:
            return tensor[index]

    @staticmethod
    def load_batch(batch):
        return tuple(map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, batch))

    # Fetch rule embeddings, either from buffer or by re-calculating
    def rule_embed(self, force=False):
        if not force and not self.arg('param_relation_embed'):
            return self._rule_embed

        relation_embed = self.rotate._attatch_empty_relation()
        rule_embed = torch.zeros(self.num_rule, self.rotate.embed_dim).cuda()
        for i in range(self.MAX_RULE_LEN):
            rule_embed += self.index_select(relation_embed, self.rules[i])
        return rule_embed

    # Init rules
    def set_rules(self, rules):
        paths = rules
        r = self.r
        self.eval()

        # self.MAX_RULE_LEN = 0
        # for path in rules:
        # 	self.MAX_RULE_LEN = max(self.MAX_RULE_LEN, len(path))
        self.MAX_RULE_LEN = self.arg('max_rule_len')

        pad = self.R - 1
        gen_end = pad
        gen_pad = self.R
        rules = []
        rules_gen = []
        rules_exp = []

        for path in paths:
            npad = (self.MAX_RULE_LEN - len(path))
            rules.append(path + (pad,) * npad)
            rules_gen.append((r,) + path + (gen_end,) + (gen_pad,) * npad)
            rules_exp.append(tuple(path))

        self.rules = torch.LongTensor(rules).t().cuda()
        # print(self.rules.size())
        self.rules_gen = torch.LongTensor(rules_gen).cuda()
        self.rules_exp = tuple(rules_exp)

    @property
    def num_rule(self):
        return self.rules.size(1)

    # Finding pseudo-groundings for a specific (h, r)
    def pgnd(self, h, i, num=None, rgnd=None):
        if num is None:
            num = self.arg('pgnd_num')

        key = (h, self.r, tuple(self.rules_exp[i]))
        if key in self.pgnd_buffer:
            return self.pgnd_buffer[key]

        with torch.no_grad():
            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed[i])
            if ans_can is None:
                dist = self.rotate.dist(rule_embed, self.rotate.entity_embed)
            else:
                can_dist = self.rotate.dist(rule_embed, self.rotate.entity_embed[ans_can])
                dist = torch.zeros(self.E).cuda() + 1e10
                dist[ans_can] = can_dist

            if rgnd is not None:
                # print(len(rgnd), dist.size())
                dist[torch.LongTensor(rgnd).cuda()] = 1e10
            ret = torch.arange(self.E).cuda()[dist <= self.rotate.gamma]

            dist[ret] = 1e10
            num = min(num, dist.size(0) - len(rgnd)) - ret.size(-1)
            if num > 0:
                tmp = dist.topk(num, dim=0, largest=False, sorted=False)[1]
                ret = torch.cat([ret, tmp], dim=0)

        self.pgnd_buffer[key] = ret
        ##########
        # print(h, sorted(ret.cpu().numpy().tolist()))
        return ret

    # Calculate score in formula 17. A sparse matrix is given with column_idx=crule, row_idx=centity. Returns score in (17) in paper, as the value of the sparse matrix.
    def cscore(self, rule_embed, crule, centity, cweight):
        score = self.rotate.compare(rule_embed, self.rotate.entity_embed, crule, centity)
        score = (self.rotate.gamma - score).sigmoid()
        if self.arg('drop_neg_gnd'):
            score = score * (score >= 0.5)
        score = score * cweight
        return score

    # Returns the rule's value in (16)
    def rule_value(self, batch, weighted=False):
        num_rule = self.num_rule
        h, t_list, mask, crule, centity, cweight = self.load_batch(batch)
        with torch.no_grad():

            rule_embed = self.rotate.embed(h, self.tmp__rule_embed)
            cscore = self.cscore(rule_embed, crule, centity, cweight)

            indices = torch.stack([crule, centity], 0)

            def cvalue(cscore):
                if cscore.size(0) == 0:
                    return torch.zeros(num_rule).cuda()
                return torch.sparse.sum(torch.sparse.FloatTensor(
                    indices,
                    cscore,
                    torch.Size([num_rule, self.E])
                ).cuda(), -1).to_dense()


            pos = cvalue(cscore * mask[centity])
            neg = cvalue(cscore * ~mask[centity])
            score = cvalue(cscore)
            num = cvalue(cweight).clamp(min=0.001)

            pos_num = cvalue(cweight * mask[centity]).clamp(min=0.001)
            neg_num = cvalue(cweight * ~mask[centity]).clamp(min=0.001)


            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            value = self.arg('rule_value_def', apply=eval_ctx(locals()))

            if weighted:
                value *= len(t_list)

            if hasattr(self, 'tmp__rule_value'):
                self.tmp__rule_value += value
                self.tmp__num_init += len(t_list)

        return value

    # Choose rules, which has top `num_samples` of `value` and has a non negative `nonneg`
    def choose_rules(self, value, nonneg=None, num_samples=None, return_mask=False):
        if num_samples is None:
            num_samples = self.arg('max_best_rules')
        ################
        # print(f"choose_rules num = {num_samples}")
        with torch.no_grad():
            num_rule = value.size(-1)
            topk = value.topk(min(num_samples - 1, num_rule), dim=0, largest=True, sorted=False)[1]
            cho = torch.zeros(num_rule).bool().cuda()
            cho[topk] = True
            if nonneg is not None:
                cho[nonneg < 0] = False

        if return_mask:
            return cho
        return mask2list(cho)

    # Choose best rules for each batch, for M-step
    def best_rules(self, batch, num_samples=None):
        with torch.no_grad():
            w = self.rule_value(batch)
            value = (w + self.arg('prior_coef') * self.prior) * self.rule_weight
            cho = self.choose_rules(value, nonneg=w, num_samples=num_samples)
        return cho

    # For a new relation, init rule weights and choose rules
    def relation_init(self, r=None, rule_file=None, rules=None, prior=None, force_init_weight=False):
        print = self.print
        if r is not None:
            self.r = r
        r = self.r
        if rules is None:
            assert rule_file is not None
            rules = [((r,), 1, -1)]
            rule_set = set([tuple(), (r,)])
            with open(rule_file) as file:
                for i, line in enumerate(file):
                    try:
                        path, prec = line.split('\t')
                        path = tuple(map(int, path.split()))
                        prec = float(prec.split()[0])

                        if not (prec >= 0.0001):
                            # to avoid negative and nan
                            prec = 0.0001

                        if path in rule_set:
                            continue
                        rule_set.add(path)
                        if len(path) <= self.arg('max_rule_len'):
                            rules.append((path, prec, i))
                    except:
                        continue

            rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)[:self.arg('max_beam_rules')]
            print(f"Loaded from file: |rules| = {len(rules)} max_rule_len = {self.arg('max_rule_len')}")
            x = torch.tensor([prec for _, prec, _ in rules]).cuda()
            prior = -torch.log((1 - x.float()).clamp(min=1e-6))
            # prior = x
            rules = [path for path, _, _ in rules]
        else:
            assert prior is not None

        self.prior = prior
        self.set_rules(rules)

        num_rule = self.num_rule
        with torch.no_grad():
            self.tmp__rule_value = torch.zeros(num_rule).cuda()
            self.tmp__rule_embed = self.rule_embed(force=True).detach()
            self.tmp__num_init = 0
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        init_weight = force_init_weight or not self.arg('init_weight_with_prior')

        if init_weight:
            for batch in self.make_batchs(init=True):
                self.rule_value(batch, weighted=True)

        with torch.no_grad():
            # self.tmp__rule_value[torch.isnan(self.tmp__rule_value)] = 0
            value = self.tmp__rule_value / max(self.tmp__num_init, 1) + self.arg('prior_coef') * self.prior
            nonneg = self.tmp__rule_value
            if self.arg('use_neg_rules') or not init_weight:
                nonneg = None
            cho = self.choose_rules(value, num_samples=self.arg('max_rules'), nonneg=nonneg, return_mask=True)

            cho[0] = True
            cho_list = mask2list(cho).detach().cpu().numpy().tolist()
            value_list = value.detach().cpu().numpy().tolist()
            cho_list = sorted(cho_list,
                              key=lambda x: (x == 0, value_list[x]), reverse=True)
            assert cho_list[0] == 0
            cho = torch.LongTensor(cho_list).cuda()

            value = value[cho]
            self.tmp__rule_value = self.tmp__rule_value[cho]
            self.prior = self.prior[cho]
            self.rules = self.rules[:, cho]
            self.rules_gen = self.rules_gen[cho]
            self.rules_exp = [self.rules_exp[x] for x in cho_list]

        if init_weight:
            weight = self.tmp__rule_value
        else:
            weight = self.prior

        print(f"weight_init: num = {self.num_rule} [{weight.min().item()}, {weight.max().item()}]")
        weight = weight.clamp(min=0.0001)
        weight /= weight.max()
        weight[0] = 1.0
        self.rule_weight_raw = torch.nn.Parameter(weight)

        del self.tmp__rule_value
        del self.tmp__num_init

        with torch.no_grad():
            self.tmp__rule_embed = self.rule_embed(force=True).detach()
            if not self.arg('param_relation_embed'):
                self._rule_embed = self.tmp__rule_embed

        self.make_batchs()

        del self.tmp__rule_embed

    # Default arguments for predictor
    def set_args(self, args):
        self._args = dict()
        def_args = dict()
        def_args['rotate_pretrained'] = None
        def_args['max_beam_rules'] = 3000
        def_args['max_rules'] = 1000
        def_args['max_rule_len'] = 4
        def_args['max_h'] = 5000
        def_args['max_best_rules'] = 300
        def_args['param_relation_embed'] = True
        def_args['param_entity_embed'] = False
        def_args['init_weight_with_prior'] = False
        def_args['prior_coef'] = 0.01
        def_args['use_neg_rules'] = False
        def_args['disable_gnd'] = False
        def_args['disable_selflink'] = False
        def_args['drop_neg_gnd'] = False
        def_args['pgnd_num'] = 256
        def_args['pgnd_selflink_rate'] = 8
        def_args['pgnd_nonselflink_rate'] = 0
        def_args['pgnd_weight'] = 0.1
        def_args['max_pgnd_rules'] = None  # def_args['max_rules']
        def_args['predictor_num_epoch'] = 200000
        def_args['predictor_early_break_rate'] = 1 / 5
        def_args['predictor_lr'] = 5e-5
        def_args['predictor_batch_size'] = 1
        def_args['predictor_print_epoch'] = 50
        def_args['predictor_init_print_epoch'] = 10
        def_args['predictor_valid_epoch'] = 100
        def_args['predictor_eval_rate'] = 4
        def_args['rule_value_def'] = '(pos - neg) / num'
        def_args['metrics_score_def'] = '(mrr+0.9*h1+0.8*h3+0.7*h10+0.01/max(1,mr), mrr, mr, h1, h3, h10, -mr)'
        def_args['answer_candidates'] = None
        def_args['record_test'] = False

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in def_args.items():
            self._args[k] = str(v) if make_str(v) else v
        for k, v in args.items():
            # if k not in self._args:
            # 	print(f"Warning: Unused argument '{k}'")
            self._args[k] = str(v) if make_str(v) else v

    def arg(self, name, apply=None):
        # print(self._args[name])
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

    @property
    def rule_weight(self):
        return self.rule_weight_raw

    def forward(self, batch):

        E = self.E
        R = self.R

        rule_weight = self.rule_weight
        _rule_embed = self.rule_embed()
        rule_embed = []
        crule = []
        crule_weight = []
        centity = []
        cweight = []
        csplit = [0]

        for single in batch:
            _h, _, _, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) == 0:
                csplit.append(csplit[-1])
                continue
            crule.append(_crule + len(rule_embed) * self.num_rule)
            crule_weight.append(rule_weight.index_select(0, _crule))
            centity.append(_centity)
            cweight.append(_cweight)
            rule_embed.append(self.rotate.embed(_h, _rule_embed))
            csplit.append(csplit[-1] + _crule.size(-1))


        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            crule_weight = torch.tensor([]).cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).cuda()
            rule_embed = torch.tensor([]).cuda()
            cscore = torch.tensor([]).cuda()
        else:
            crule = torch.cat(crule, dim=0)
            crule_weight = torch.cat(crule_weight, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)
            rule_embed = torch.cat(rule_embed, dim=0)
            cscore = self.cscore(rule_embed, crule, centity, cweight) * crule_weight




        loss = torch.tensor(0.0).cuda().requires_grad_() + 0.0
        result = []

        for i, single in enumerate(batch):
            _h, t_list, mask, _crule, _centity, _cweight = self.load_batch(single)
            if _crule.size(0) != 0:
                crange = torch.arange(csplit[i], csplit[i + 1]).cuda()
                score = torch.sparse.sum(torch.sparse.FloatTensor(
                    torch.stack([_centity, _crule], dim=0),
                    self.index_select(cscore, crange),
                    torch.Size([E, self.num_rule])
                ), -1).to_dense()
            else:
                score = torch.zeros(self.E).cuda()
                score.requires_grad_()

            ans_can = self.arg('answer_candidates', apply=lambda x: x)
            if ans_can is not None:
                ans_can = ans_can.cuda()
                score = self.index_select(score, ans_can)
                mask = self.index_select(mask, ans_can)

                map_arr = -torch.ones(self.E).long().cuda()
                map_arr[ans_can] = torch.arange(ans_can.size(0)).long().cuda()
                map_arr = map_arr.detach().cpu().numpy().tolist()
                map_fn = lambda x: map_arr[x]
                t_list = list(map(map_fn, t_list))

            if self.recording:
                self.record.append((score.cpu(), mask, t_list))

            elif not self.training:
                for t in t_list:
                    result.append(self.metrics.apply(score, mask.bool(), t))

            if score.dim() == 0:
                continue

            score = score.softmax(dim=-1)
            neg = score.masked_select(~mask.bool())

            loss += neg.sum()

            for t in t_list:
                s = score[t]
                wrong = (neg > s)
                loss += ((neg - s) * wrong).sum() / wrong.sum().clamp(min=1)

        return loss / len(batch), self.metrics.summary(result)

    def _evaluate(self, valid_batch, batch_size=None):
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')
        # print(print_epoch)

        self.eval()
        with torch.no_grad():
            result = Metrics.zero_value()
            for i in range(0, len(valid_batch), batch_size):
                cur = model(valid_batch[i: i + batch_size])[1]
                result = Metrics.merge(result, cur)
                if i % print_epoch == 0 and i > 0:
                    print(f"eval #{i}/{len(valid_batch)}")
        return result

    # make a single batch, find groundings and pseudo-groundings
    def _make_batch(self, h, t_list, answer=None, rgnd_buffer=None):
        # print("make_batch in")
        if answer is None:
            answer = t_list
        if rgnd_buffer is None:
            rgnd_buffer = self.rgnd_buffer
        crule = []
        centity = []
        cweight = []
        gnd = []
        max_pgnd_rules = self.arg('max_pgnd_rules')
        if max_pgnd_rules is None:
            max_pgnd_rules = self.arg('max_rules')
        for i, rule in enumerate(self.rules_exp):
            # print(f"iter i = {i} / {len(self.rules_exp)}")
            if i != 0 and not self.arg('disable_gnd'):
                key = (h, self.r, rule)
                if key in rgnd_buffer:
                    rgnd = rgnd_buffer[key]
                else:
                    # print("gnd in")
                    rgnd = groundings.groundings(h, rule)

                    ans_can = self.arg('answer_candidates', apply=lambda x: x)
                    if ans_can is not None:
                        ans_can = set(ans_can.cpu().numpy().tolist())
                        rgnd = list(filter(lambda x: x in ans_can, rgnd))
                    rgnd_buffer[key] = rgnd

                ones = torch.ones(len(rgnd))
                centity.append(torch.LongTensor(rgnd))
                crule.append(ones.long() * i)
                cweight.append(ones)
            else:
                rgnd = []

            gnd.append(rgnd)
            if i == 0 and self.arg('disable_selflink'):
                continue
            if i >= max_pgnd_rules:
                continue
            num = self.arg('pgnd_num') * self.arg('pgnd_selflink_rate' if i == 0 else 'pgnd_nonselflink_rate')
            pgnd = self.pgnd(h, i, num, gnd[i])

            ones = torch.ones(len(pgnd))
            centity.append(pgnd.long().cpu())
            crule.append(ones.long() * i)
            cweight.append(ones * self.arg('pgnd_weight'))

        # print("iter done")
        if len(crule) == 0:
            crule = torch.tensor([]).long().cuda()
            centity = torch.tensor([]).long().cuda()
            cweight = torch.tensor([]).float().cuda()
        else:
            crule = torch.cat(crule, dim=0)
            centity = torch.cat(centity, dim=0)
            cweight = torch.cat(cweight, dim=0)

        #################
        # print("work", answer)

        # print("make_batch out")

        return h, t_list, list2mask(answer, self.E), crule, centity, cweight

    # make all batchs
    def make_batchs(self, init=False):
        print = self.print
        # if r is not None:
        # 	self.r = r
        dataset = self.dataset
        graph = build_graph(dataset['train'], self.E, self.R)
        graph_test = build_graph(dataset['train'] + dataset['valid'], self.E, self.R)

        def filter(tri):
            a = defaultdict(lambda: [])
            for h, r, t in tri:
                if r == self.r:
                    a[h].append(t)
            return a

        train = filter(dataset['train'])
        valid = filter(dataset['valid'])
        test = filter(dataset['test'])

        answer_valid = defaultdict(lambda: [])
        answer_test = defaultdict(lambda: [])
        for a in [train, valid]:
            for k, v in a.items():
                answer_valid[k] += v
                answer_test[k] += v
        for k, v in test.items():
            answer_test[k] += v

        if len(train) > self.arg('max_h'):
            from random import shuffle
            train = list(train.items())
            shuffle(train)
            train = train[:self.arg('max_h')]
            train = {k: v for (k, v) in train}

        print_epoch = self.arg('predictor_init_print_epoch')

        self.train_batch = []
        self.valid_batch = []
        self.test_batch = []

        groundings.use_graph(graph)

        if init:
            def gen_init(self, train, print_epoch):
                for i, (h, t_list) in enumerate(train.items()):
                    if i % print_epoch == 0:
                        print(f"init_batch: {i}/{len(train)}")
                    yield self._make_batch(h, t_list)

            return gen_init(self, train, print_epoch)

        for i, (h, t_list) in enumerate(train.items()):
            if i % print_epoch == 0:
                print(f"train_batch: {i}/{len(train)}")
            batch = list(self._make_batch(h, t_list))
            for t in t_list:
                batch[1] = [t]
                self.train_batch.append(tuple(batch))

        for i, (h, t_list) in enumerate(valid.items()):
            if i % print_epoch == 0:
                print(f"valid_batch: {i}/{len(valid)}")
            self.valid_batch.append(self._make_batch(h, t_list, answer=answer_valid[h]))

        groundings.use_graph(graph_test)
        for i, (h, t_list) in enumerate(test.items()):
            if i % print_epoch == 0:
                print(f"test_batch: {i}/{len(test)}")
            self.test_batch.append(
                self._make_batch(h, t_list, answer=answer_test[h], rgnd_buffer=self.rgnd_buffer_test))

    # Make batchs for generator, used in M-step
    def make_gen_batch(self, generator_version=1):
        self.tmp__rule_embed = self.rule_embed()
        weight = torch.zeros_like(self.rule_weight_raw).long().cuda()
        for i, batch in enumerate(self.train_batch):
            cho = self.best_rules(batch)
            weight[cho] += len(batch[1])  # len(t_list)

        nonzero = (weight > 0)
        rules = self.rules_gen[nonzero]
        weight = weight[nonzero]

        if generator_version >= 2:
            rules = rules[:, 1:]
            return self.r, rules, weight

        inp = rules[:, :-1]
        tar = rules[:, 1:]
        gen_pad = self.R
        mask = (tar != gen_pad)
        del self.tmp__rule_embed
        return inp, tar, mask, weight

    def train_model(self):
        # self.make_batchs()
        train_batch = self.train_batch
        valid_batch = self.valid_batch
        test_batch = self.test_batch
        model = self
        print = self.print
        batch_size = self.arg('predictor_batch_size')
        num_epoch = self.arg('predictor_num_epoch')  # / batch_size
        lr = self.arg('predictor_lr')  # * batch_size
        print_epoch = self.arg('predictor_print_epoch')
        valid_epoch = self.arg('predictor_valid_epoch')

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 5)

        self.best = Metrics.init_value()
        self.best_model = self.state_dict()

        def train_step(batch):
            self.train()
            loss, _ = self(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            return loss

        def metrics_score(result):
            result = Metrics.pretty(result)
            mr = result['mr']
            mrr = result['mrr']
            h1 = result['h1']
            h3 = result['h3']
            h10 = result['h10']

            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            return self.arg('metrics_score_def', apply=eval_ctx(locals()))

        def format(result):
            s = ""
            for k, v in Metrics.pretty(result).items():
                if k == 'num':
                    continue
                s += k + ":"
                s += "%.4lf " % v
            return s

        def valid():
            result = self._evaluate(valid_batch)
            updated = False
            if metrics_score(result) > metrics_score(self.best):
                updated = True
                self.best = result
                self.best_model = copy.deepcopy(self.state_dict())
            print(f"valid = {format(result)} {'updated' if updated else ''}")
            return updated, result

        last_update = 0
        cum_loss = 0
        valid()

        relation_embed_init = self.rotate.relation_embed.clone()

        if len(train_batch) == 0:
            num_epoch = 0
        for epoch in range(1, num_epoch + 1):
            if epoch % max(1, len(train_batch) // batch_size) == 0:
                from random import shuffle
                shuffle(train_batch)
            batch = [train_batch[(epoch * batch_size + i) % len(train_batch)] for i in range(batch_size)]
            loss = train_step(batch)
            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_predictor #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss *= 0

            if epoch % valid_epoch == 0:
                if valid()[0]:
                    last_update = epoch
                elif epoch - last_update >= self.arg('predictor_early_break_rate') * num_epoch:
                    print(f"Early break: Never updated since {last_update}")
                    break
                if 1 - 1e-6 < Metrics.pretty(self.best)['mr'] < 1 + 1e-6:
                    print(f"Early break: Perfect")
                    break

        with torch.no_grad():
            self.load_state_dict(self.best_model)
            self.rotate.relation_embed *= 0
            self.rotate.relation_embed += relation_embed_init
            self.rule_weight_raw[0] += 1000.0
            valid()

        self.load_state_dict(self.best_model)
        best = self.best
        if self.arg('record_test'):
            backup = self.recording
            self.record = []
            self.recording = True
        test = self._evaluate(test_batch)
        if self.arg('record_test'):
            self.recording = backup

        print("__V__\t" + ("\t".join([str(self.r), str(int(best[0]))] + list(map(lambda x: "%.4lf" % x, best[1:])))))
        print("__T__\t" + ("\t".join([str(self.r), str(int(test[0]))] + list(map(lambda x: "%.4lf" % x, test[1:])))))

        return best, test


class RNNLogicGenerator(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, hidden_dim, print=print):
        super(RNNLogicGenerator, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mov = num_relations // 2
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = num_relations
        self.padding_idx = self.num_relations + 1
        self.num_layers = 1
        self.use_cuda = True
        self.print = print

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.rnn = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.print = print

        self.cuda()

    def inv(self, r):
        if r < self.mov:
            return r + self.mov
        else:
            return r - self.mov

    def zero_state(self, batch_size):
        state_shape = (self.num_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)
        outputs, hidden = self.rnn(embedding, hidden)
        logits = self.linear(outputs)
        # Predictor.clean()
        return logits, hidden

    def loss(self, inputs, target, mask, weight):
        if self.use_cuda:
            inputs = inputs.cuda()
            target = target.cuda()
            mask = mask.cuda()
            weight = weight.cuda()

        hidden = self.zero_state(inputs.size(0))
        logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss

    def sample(self, relation):
        rule = [relation]
        relation = torch.LongTensor([relation])
        if self.use_cuda:
            relation = relation.cuda()
        hidden = self.zero_state(1)
        while True:
            inputs = torch.LongTensor([[rule[-1]]])
            if self.use_cuda:
                inputs = inputs.cuda()
            logits, hidden = self.forward(inputs, relation, hidden)
            probability = torch.softmax(logits.squeeze(0).squeeze(0), dim=-1)
            sample = torch.multinomial(probability, 1).item()
            if sample == self.ending_idx:
                break
            rule.append(sample)
        return rule

    def log_probability(self, rule):
        rule.append(self.ending_idx)
        relation = torch.LongTensor([rule[0]])
        if self.use_cuda:
            relation = relation.cuda()
        hidden = self.zero_state(1)
        log_prob = 0.0
        for k in range(1, len(rule)):
            inputs = torch.LongTensor([[rule[k - 1]]])
            if self.use_cuda:
                inputs = inputs.cuda()
            logits, hidden = self.forward(inputs, relation, hidden)
            log_prob += torch.log_softmax(logits.squeeze(0).squeeze(0), dim=-1)[rule[k]]
        return log_prob

    def next_relation_log_probability(self, seq):
        inputs = torch.LongTensor([seq])
        relation = torch.LongTensor([seq[0]])
        if self.use_cuda:
            inputs = inputs.cuda()
            relation = relation.cuda()
        hidden = self.zero_state(1)
        logits, hidden = self.forward(inputs, relation, hidden)
        log_prob = torch.log_softmax(logits[0, -1, :] * 5, dim=-1).data.cpu().numpy().tolist()
        return log_prob

    def train_model(self, gen_batch, num_epoch=10000, lr=1e-3, print_epoch=100):
        print = self.print
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 10)

        cum_loss = 0
        if gen_batch[0].size(0) == 0:
            num_epoch = 0
        for epoch in range(1, num_epoch + 1):

            loss = self.loss(*gen_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_generator #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss = 0

    def beam_search(self, relation, num_samples, max_len):
        max_len += 1
        print = self.print
        with torch.no_grad():
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                self.print(f"k = {k} |prev| = {len(prev_rules)}")
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != self.ending_idx
                    log_prob = self.next_relation_log_probability(rule)
                    for i in (range(self.label_size) if (k + 1) != max_len else [self.ending_idx]):
                        # if k != 0 and rule[-1] == self.inv(i):
                        # 	continue
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != self.ending_idx else found_rules).append((new_rule, new_score))

                # Predictor.clean()
                # if _i % 100 == 0:
                # 	pass
                # self.print(f"beam_search k = {k} i = {_i}")
                prev_rules = sorted(current_rules, key=lambda x: x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x: x[1], reverse=True)[:num_samples]

            self.print(f"beam_search |rules| = {len(found_rules)}")
            ret = ((rule[1:-1], score) for rule, score in found_rules)
            return ret
