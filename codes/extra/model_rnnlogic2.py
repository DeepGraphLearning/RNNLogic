import copy
import gc
from collections import defaultdict

import groundings
from knowledge_graph_utils import *
from metrics import Metrics
from reasoning_model import ReasoningModel
from rotate import RotatE
from model_rnnlogic import RNNLogicPredictor
import random


def make_dicts(dataset, label):
    ret = dict()
    ret['E'] = dataset['E']
    ret['R'] = dataset['R']
    ret['r'] = set()
    ret['tri'] = dataset[label]
    ret['h'] = defaultdict(lambda: set())
    ret['ht'] = defaultdict(lambda: [])
    ret['t'] = defaultdict(lambda: [])
    for (h, r, t) in dataset[label]:
        ret['h'][r].add(h)
        ret['ht'][r].append((h, t))
        ret['t'][(h, r)].append(t)
        ret['r'].add(r)

    ret['r'] = list(ret['r'])
    return ret


class RNNLogic2(torch.nn.Module):
    def __init__(self, dataset, args, print=print):
        super(RNNLogic2, self).__init__()
        self.E = dataset['E']
        self.R = dataset['R']
        assert self.R % 2 == 0
        self.set_args(args)

        self.dataset = dataset
        self._print = print
        self.print = self.log_print
        self.predictor_init = lambda: RNNLogicPredictor(self.dataset, self._args, print=self.log_print)
        self.generator = RNNLogicGenerator2(self.dataset, self.arg('generator_embed_dim'), self.arg('generator_hidden_dim'),
                                           rotate_pretrained=self.arg('rotate_pretrained', apply=str),
                                           print=self.log_print)

        print = self.print
        print("RNNLogic Init", self.E, self.R)

    # self.predictor_init = self.predictor
    # del self.predictor

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

    def pretrain_generator(self, model_file='generator_pretrained.pth', load=False, rule_file_map=None):
        if load:
            self.generator.load_state_dict(torch.load(model_file))
            return
        assert rule_file_map is not None

        dicts = make_dicts(self.dataset, 'train')
        pretrain_batch = []

        self.MAX_RULE_LEN = self.arg('max_rule_len')

        for r in range(self.R):
            rule_file = rule_file_map(r)
            rule_set = set()
            rules = []
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
                        if len(path) <= self.MAX_RULE_LEN:
                            npad = self.MAX_RULE_LEN - len(path)
                            rule = path + (self.generator.end_idx,) + (self.generator.pad_idx,) * npad
                            rules.append((rule, prec, i))
                    except:
                        continue
            rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)[:self.arg('max_beam_rules')]
            batch = (
                r,
                torch.tensor([rule for rule, _, _ in rules]).cuda(),
                torch.tensor([weight for _, weight, _ in rules]).cuda()
            )
            print(f"pretrain r = {r} shape = {batch[1].size()} {batch[2].size()}")
            for _ in dicts['ht'][r]:
                pretrain_batch.append(batch)

        lr = self.arg('generator_pretrain_lr')
        num_epoch = self.arg('generator_pretrain_epoch')
        print_epoch = self.arg('generator_print_epoch')
        opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch)
        cum_loss = 0

        for epoch in range(1, num_epoch + 1):
            if epoch % len(pretrain_batch) == 1:
                random.shuffle(pretrain_batch)

            r, rules, weights = pretrain_batch[epoch % len(pretrain_batch)]
            # print(r)
            # print(rules)
            # print(weights)

            loss = self.generator.loss(r, rules, weights)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            cum_loss += loss.item()
            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"pretrain_generator #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss = 0

            if epoch % (print_epoch * 100) == 0:
                torch.save(self.generator.state_dict(), model_file)
                print("Saved!")
        torch.save(self.generator.state_dict(), model_file)

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

        for self.em in range(num_em_epoch + 1):
            self.predictor = self.predictor_init()
            self.predictor.pgnd_buffer = pgnd_buffer
            self.predictor.rgnd_buffer = rgnd_buffer
            self.predictor.rgnd_buffer_test = rgnd_buffer_test

            # No difference between #0 and others
            # if self.em == 0:
            #     self.predictor.relation_init(r=r, rule_file=rule_file, force_init_weight=self.arg('init_weight_boot'))
            # else:
            sampled = set()
            sampled.add((r,))
            sampled.add(tuple())

            rules = [(r,)]
            prior = [0.0, ]
            for rule, score in self.generator.beam_search(self.generator.embed_r[r],
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

            valid, test = self.predictor.train_model()

            if self.em != self.num_em_epoch:
                gen_batch = self.predictor.make_gen_batch(generator_version=2)
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

        return valid, test

    def arg(self, name, apply=None):
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)

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
        def_args['generator_pretrain_epoch'] = 300000
        def_args['generator_pretrain_lr'] = 1e-4
        def_args['generator_num_epoch'] = 10000
        def_args['generator_print_epoch'] = 100
        def_args['init_weight_boot'] = False
        def_args['rotate_pretrained'] = None
        def_args['max_rule_len'] = 3

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


class RNNLogicGenerator2(ReasoningModel):
    def __init__(self, dataset, embed_dim, hidden_dim, rotate_pretrained=None, print=print):
        super(RNNLogicGenerator2, self).__init__()
        self.print = print
        self.E = dataset['E']
        self.R = dataset['R']
        self.end_idx = self.R
        self.pad_idx = self.R + 1
        self.vocab_size = self.R + 2
        self.dataset = dataset
        # self.metrics = metrics.Metrics(E)

        self.rotate = RotatE(dataset, rotate_pretrained)
        # self.rotate.enable_parameter('relation_embed')
        # self.rotate.enable_parameter('entity_embed')
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.softmax_coef = 1
        self.beam_coef = 3

        self.embed = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)

        self.to_embed = torch.nn.Linear(self.rotate.embed_dim * 2, self.embed_dim)

        self.rnn = torch.nn.LSTM(self.embed_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.to_logits = torch.nn.Linear(self.hidden_dim, self.vocab_size - 1)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.cuda()

        train_ht = make_dicts(self.dataset, 'train')['ht']
        # self.embed_r = []
        self.embed_r = self.rotate.relation_embed
        self.embed_r = torch.cat([torch.cos(self.embed_r), torch.sin(self.embed_r)], dim=-1)
        # with torch.no_grad():
        #     for r in range(self.R):
        #         print(f"Calc embed_r r = {r}")
        #         self.embed_r.append(self.average_embed(train_ht[r]))
        #     self.embed_r = torch.stack(self.embed_r, dim=0)

    def average_embed(self, ht_list):
        a = torch.zeros(self.rotate.embed_dim * 2).cuda()
        for (h, t) in ht_list:
            re_h, im_h = torch.chunk(self.rotate.entity_embed[h].cuda(), 2, dim=-1)
            re_t, im_t = torch.chunk(self.rotate.entity_embed[t].cuda(), 2, dim=-1)
            dom = im_h.pow(2) + re_h.pow(2)
            re_r = (re_h * re_t + im_h * im_t) / dom
            im_r = (re_h * im_t - im_h * re_t) / dom
            a += torch.cat([re_r, im_r], dim=-1)
        a = a / max(1, len(ht_list))
        assert torch.isnan(a).sum().item() == 0
        return a

    def state_begin(self, batch_size=1):
        tmp = torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()
        return (tmp, tmp)

    def forward(self, relations, inputs):
        batch_size = inputs.size(0)
        if isinstance(relations, int):
            embed_r = self.embed_r[relations].expand(batch_size, -1)
        elif isinstance(relations, torch.LongTensor):
            embed_r = self.embed_r.index_select(0, relations).cuda()
        else:
            embed_r = relations.cuda().expand(batch_size, -1)

        embed_r = self.to_embed(embed_r)

        embed_i = self.embed(inputs)
        embed_r = embed_r.unsqueeze(dim=1)
        embed_i = torch.cat([embed_r, embed_i], dim=1)
        embed_r = embed_r.expand(-1, embed_i.size(1), -1)
        # print(embed_i.size(), embed_r.size())

        embed = torch.cat([embed_i, embed_r], dim=-1)

        out, _ = self.rnn(embed, self.state_begin(batch_size))
        logits = self.to_logits(out)

        return logits

    def loss(self, relations, rules, weight=None):
        rules = rules.cuda()
        inputs = rules[:, :-1]
        target = rules
        mask = (target != self.pad_idx)
        logits = self(relations, inputs)
        logits = logits.masked_select(mask.unsqueeze(-1)).view(-1, self.vocab_size - 1)
        target = target.masked_select(mask)
        cri = self.criterion(logits, target)
        if weight is None:
            return cri.sum() / cri.size(0)
        weight = (mask.t() * weight).t().masked_select(mask)
        # print(f"Generator loss weighted = {(cri * weight).sum()}")
        return (cri * weight).sum() / weight.sum()

    def beam_search(self, embed_r, num_samples, max_len):
        max_len += 1
        with torch.no_grad():
            rules = []
            prev_prefix = [[tuple(), torch.tensor(0.0).cpu()]]
            limit = None
            for k in range(max_len):
                prefix = []
                for _i, (rule, score) in enumerate(prev_prefix):
                    inputs = torch.LongTensor([rule]).cuda()
                    relations = embed_r.cuda().unsqueeze(dim=0)
                    logits = self(relations, inputs)
                    log_prob = torch.log_softmax(logits[0, -1, :] * 5, dim=-1)
                    log_prob = log_prob.cpu()

                    ran = range(self.vocab_size - 1) if (k + 1 != max_len) else [self.end_idx]
                    for i in ran:
                        new_score = score + log_prob[i]
                        if limit is not None and new_score < limit:
                            continue
                        if i == self.end_idx:
                            if len(rule) > 0:
                                rules.append((rule, new_score))
                        else:
                            prefix.append((rule + (i,), new_score))

                prev_prefix = sorted(prefix, key=lambda x: x[1], reverse=True)[:num_samples]
                rules = sorted(rules, key=lambda x: x[1], reverse=True)[:num_samples]
                if len(rules) > 0:
                    limit = rules[-1][1]

        # _r = torch.LongTensor([rule for rule, score in rules]).cuda()
        # _s = torch.LongTensor([score for rule, score in rules]).cuda().softmax(dim=-1)
        return rules

    def train_model(self, gen_batch, num_epoch=10000, lr=1e-3, print_epoch=100):
        print = self.print
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 10)

        cum_loss = 0
        if gen_batch[1].size(0) == 0:
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
