import torch

class Generator(torch.nn.Module):
    def __init__(self, graph, num_layers, embedding_dim, hidden_dim):
        super(Generator, self).__init__()
        self.graph = graph
        self.num_relations = graph.relation_size

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = self.num_relations
        self.padding_idx = self.num_relations + 1

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.rnn = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)
        outputs, hidden = self.rnn(embedding, hidden)
        logits = self.linear(outputs)
        return logits, hidden

    def loss(self, inputs, target, mask, weight, hidden):
        logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss

    