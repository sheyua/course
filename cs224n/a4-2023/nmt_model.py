#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""
from typing import List, Tuple
from torch import Tensor, float32
from torch.nn import Module
from collections import namedtuple
from vocab import Vocab
from utils import SentType, SentsType
from model_embeddings import ModelEmbeddings


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
DecStateType = Tuple[Tensor, Tensor]


class NMT(Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    @property
    def dtype(self) -> type: return float32

    def __init__(self, embed_size: int, hidden_size: int, vocab: Vocab, dropout_rate: float=0.2) -> None:
        """ Init NMT Model.

        @param embed_size: Embedding size (dimensionality)
        @param hidden_size: Hidden Size, the size of hidden states (dimensionality)
        @param vocab: Vocabulary object containing src and tgt languages See vocab.py for documentation.
        @param dropout_rate: Dropout probability, for attention
        """
        from torch.nn import Conv1d, LSTM, LSTMCell, Linear, Dropout

        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size=embed_size, vocab=vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0

        # YOUR CODE HERE (~9 Lines)
        #     self.post_embed_cnn (Conv1d layer with kernel size 2, input and output channels = embed_size,
        #         padding = same to preserve output shape )
        #     self.encoder (Bidirectional LSTM with bias)
        #     self.decoder (LSTM Cell with bias)
        #     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        #     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        #     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        #     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        #     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        #     self.dropout (Dropout Layer)
        e, h, v = embed_size, self.hidden_size, len(self.vocab.tgt)
        self.post_embed_cnn = Conv1d(in_channels=e, out_channels=e, kernel_size=2, padding='same')
        self.encoder = LSTM(input_size=e, hidden_size=h, bias=True, bidirectional=True, dtype=self.dtype)
        self.decoder = LSTMCell(input_size=e + h, hidden_size=h, bias=True, dtype=self.dtype)
        self.h_projection = Linear(in_features=2 * h, out_features=h, bias=False, dtype=self.dtype)
        self.c_projection = Linear(in_features=2 * h, out_features=h, bias=False, dtype=self.dtype)
        self.att_projection = Linear(in_features=2 * h, out_features=h, bias=False, dtype=self.dtype)
        self.combined_output_projection = Linear(in_features=3 * h, out_features=h, bias=False, dtype=self.dtype)
        self.dropout = Dropout(p=self.dropout_rate, inplace=False)
        self.target_vocab_projection = Linear(in_features=h, out_features=v, bias=False, dtype=self.dtype)
        # END YOUR CODE

    def forward(self, source: SentsType, target: SentsType) -> Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source: list of source sentence tokens
        @param target: list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        from torch import gather
        from torch.nn.functional import log_softmax

        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(sents=source)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(sents=target)  # Tensor: (tgt_len, b)
        source_padded = source_padded.to(self.model_embeddings.source.weight.device)
        target_padded = target_padded.to(self.model_embeddings.source.weight.device)

        #     Run the network forward:
        #     1. Apply the encoder to `source_padded` by calling `self.encode()`
        #     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        #     3. Apply the decoder to compute combined-output by calling `self.decode()`
        #     4. Compute log probability distribution over the target vocabulary using the
        #        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        index = target_padded[1:].unsqueeze(-1)
        target_gold_words_log_prob = gather(P, index=index, dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: Tensor, source_lengths: List[int]) -> Tuple[Tensor, DecStateType]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded: Tensor of padded source sentences with shape (src_len, b), where b = batch_size,
                              src_len = maximum source sentence length. Note that these have already been sorted in
                              order of longest to shortest sentence.
        @param source_lengths: List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens: Tensor of hidden units with shape (b, src_len, h*2), where b = batch size,
                              src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state: Tuple of tensors representing the decoder's initial hidden state and cell. Both
                                 tensors should have shape (2, b, h).
        """
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        # enc_hiddens, dec_init_state = None, None

        # YOUR CODE HERE (~ 11 Lines)
        #     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        #         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        #         that there is no initial hidden state or cell for the encoder.
        #     2. Apply the post_embed_cnn layer. Before feeding X into the CNN, first use torch.permute to change the
        #         shape of X to (b, e, src_len). After getting the output from the CNN, still stored in the X variable,
        #         remember to use torch.permute again to revert X back to its original shape.
        #     3. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        #         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        #         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        #         - Note that the shape of the tensor output returned by the encoder RNN is (src_len, b, h*2) and we
        #           want to return a tensor of shape (b, src_len, h*2) as `enc_hiddens`, so you may need to do more
        #           permuting.
        #         - Note on using pad_packed_sequence -> For batched inputs, you need to make sure that each of the
        #           individual input examples has the same shape.
        #     4. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        #         - `init_decoder_hidden`:
        #             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and
        #             backwards. Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        #             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        #             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        #         - `init_decoder_cell`:
        #             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and
        #             backwards. Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        #             Apply the c_projection layer to this in order to compute init_decoder_cell.
        #             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        #

        X = self.model_embeddings.source(input=source_padded)
        total = X.size(0)
        assert total == max(source_lengths)

        # (2) post_embed_cnn layer
        X = X.permute(dims=[1, 2, 0])
        from warnings import catch_warnings, filterwarnings
        with catch_warnings():
            filterwarnings(action='ignore', category=UserWarning)
            X = self.post_embed_cnn(input=X).permute(dims=[2, 0, 1])

        # (3.1) use pack inputs to cut the hidden/cell at sentence end
        X_ = pack_padded_sequence(input=X, lengths=source_lengths, batch_first=False, enforce_sorted=False)
        o_, (h_last, c_last) = self.encoder(input=X_)
        # (3.2) use pad_packed_sequence to transform
        enc_hiddens, lengths = pad_packed_sequence(sequence=o_, batch_first=True, padding_value=0., total_length=total)

        # (4) compute init decoder state
        init_decoder_hidden = h_last.transpose(dim0=0, dim1=1).reshape([-1, 2 * self.hidden_size])
        init_decoder_hidden = self.h_projection(input=init_decoder_hidden)
        init_decoder_cell = c_last.transpose(dim0=0, dim1=1).reshape([-1, 2 * self.hidden_size])
        init_decoder_cell = self.c_projection(input=init_decoder_cell)
        dec_init_state = init_decoder_hidden, init_decoder_cell
        # END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: Tensor, enc_masks: Tensor, dec_init_state: DecStateType,
               target_padded: Tensor) -> Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens: Hidden states (b, src_len, h*2), where b = batch size,
                            src_len = maximum source sentence length, h = hidden size.
        @param enc_masks: Tensor of sentence masks (b, src_len), where b = batch size,
                          src_len = maximum source sentence length.
        @param dec_init_state: Initial state and cell for decoder
        @param target_padded: Gold-standard padded target sentences (tgt_len, b), where
                              tgt_len = maximum target sentence length, b = batch size.
        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                            tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        from torch import zeros, split, cat, stack

        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = zeros(batch_size, self.hidden_size, dtype=self.dtype, device=enc_hiddens.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = list()

        # YOUR CODE HERE (~9 Lines)
        #     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        #         which should be shape (b, src_len, h),
        #         where b = batch size, src_len = maximum source length, h = hidden size.
        #         This is applying W_{attProj} to h^enc, as described in the PDF.
        #     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        #         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        #     3. Use the torch.split function to iterate over the time dimension of Y.
        #         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        #             - Squeeze Y_t into a tensor of dimension (b, e).
        #             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
        #             - Use the step function to compute the the Decoder's next (cell, state) values
        #               as well as the new combined output o_t.
        #             - Append o_t to combined_outputs
        #             - Update o_prev to the new o_t.
        #     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        #         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        #         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        #
        # Note:
        #    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        #      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        #

        enc_hiddens_proj = self.att_projection(input=enc_hiddens)
        # Y with teacher enforcing
        Y = self.model_embeddings.target(input=target_padded)
        for y_prev in split(tensor=Y, split_size_or_sections=1):
            y_prev = y_prev.reshape([batch_size, -1])
            y_bar = cat([y_prev, o_prev], dim=1)
            dec_state, o_curr, e_curr = self.step(Ybar_t=y_bar, dec_state=dec_state, enc_hiddens=enc_hiddens,
                                                  enc_hiddens_proj=enc_hiddens_proj, enc_masks=enc_masks)
            combined_outputs.append(o_curr)
            o_prev = o_curr
            # stack
        combined_outputs = stack(tensors=combined_outputs, dim=0)
        # END YOUR CODE

        return combined_outputs

    def step(self, Ybar_t: Tensor, dec_state: DecStateType, enc_hiddens: Tensor, enc_hiddens_proj: Tensor,
             enc_masks: Tensor) -> Tuple[Tuple, Tensor, Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t: Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder, where
                       b = batch size, e = embedding size, h = hidden size.
        @param dec_state: Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size. First tensor
                          is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens: Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                            src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj: Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape
                                 (b, src_len, h), where b = batch size, src_len = maximum source length,
                                 h = hidden size.
        @param enc_masks: Tensor of sentence masks shape (b, src_len), where b = batch size, src_len is maximum source
                          length.

        @returns dec_state: Tuple of tensors both shape (b, h), where b = batch size, h = hidden size. First tensor is
                            decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output: Combined output Tensor at timestep t, shape (b, h), where b = batch size,
                                  h = hidden size.
        @returns e_t: Tensor of shape (b, src_len). It is attention scores distribution. Note: You will not use this
                      outside of this function. We are simply returning this value so that we can sanity check your
                      implementation.
        """
        from torch import bmm, cat, tanh
        from torch.nn.functional import softmax

        # combined_output = None

        # YOUR CODE HERE (~3 Lines)
        #     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        #     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        #     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        #        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        #
        #     Hints:
        #       - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        #       - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        #       - Use batched matrix multiplication (torch.bmm) to compute e_t (be careful about the shapes!)
        #       - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        #       - When using the squeeze() function make sure to specify the dimension you want to squeeze
        #           over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        batch_size = Ybar_t.size(0)
        h, c = self.decoder(input=Ybar_t, hx=dec_state)
        e_t = bmm(input=enc_hiddens_proj, mat2=h.unsqueeze(dim=2)).reshape([batch_size, -1])
        # END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # YOUR CODE HERE (~6 Lines)
        #     1. Apply softmax to e_t to yield alpha_t
        #     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        #         attention output vector, a_t.
        #     Hints:
        #           - alpha_t is shape (b, src_len)
        #           - enc_hiddens is shape (b, src_len, 2h)
        #           - a_t should be shape (b, 2h)
        #           - You will need to do some squeezing and unsqueezing.
        #     Note: b = batch size, src_len = maximum source length, h = hidden size.
        #
        #     3. Concatenate dec_hidden with a_t to compute tensor U_t
        #     4. Apply the combined output projection layer to U_t to compute tensor V_t
        #     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        alpha_t = softmax(input=e_t, dim=1)
        a_t = bmm(input=alpha_t.unsqueeze(dim=1), mat2=enc_hiddens).reshape([batch_size, -1])
        # assert a_t.shape == (batch_size, 2 * self.hidden_size)
        U_t = cat([a_t, h], dim=1)
        # assert U_t.shape == (batch_size, 3 * self.hidden_size)
        V_t = self.combined_output_projection(input=U_t)
        O_t = self.dropout(input=tanh(input=V_t))
        dec_state = h, c
        # END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: Tensor, source_lengths: List[int]) -> Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens: encodings of shape (b, src_len, 2*h), where b = batch size, src_len = max source length,
                            h = hidden size.
        @param source_lengths: List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len), where src_len = max source length,
                                     h = hidden size.
        """
        from torch import zeros

        enc_masks = zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=self.dtype)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(enc_hiddens.device)

    def beam_search(self, src_sent: SentType, beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent: a single source sentence (words)
        @param beam_size: beam size
        @param max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        from torch.nn.functional import log_softmax
        from torch import zeros, float, long, tensor, cat, topk, div

        device = self.model_embeddings.source.weight.device
        src_sents_var = self.vocab.src.to_input_tensor([src_sent]).to(device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = zeros(1, self.hidden_size, device=device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = zeros(len(hypotheses), dtype=float, device=device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=long, device=device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear,
                                                enc_masks=None)

            # log probabilities over target words
            log_p_t = log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = tensor(live_hyp_ids, dtype=long, device=device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = tensor(new_hyp_scores, dtype=float, device=device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @staticmethod
    def load(model_path: str) -> 'NMT':
        """ Load the model from a file.
        @param model_path: path to model
        """
        from torch import load

        params = load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str) -> None:
        """ Save the odel to a file.
        @param path: path to the model
        """
        from sys import stderr
        from torch import save

        print(f'save model parameters to [{path}]', file=stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        save(params, path)
