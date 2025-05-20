import tensorflow as tf



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from helper import BahdanauAttention


# ------------------ Encoder ------------------
class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers=1, cell_type='LSTM', dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=layers,
                           dropout=dropout if layers > 1 else 0.0, batch_first=True)

    def forward(self, input_seq, lengths):
        x = self.embed(input_seq)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out, hidden

# # ------------------ Attention ------------------

# ------------------ Decoder ------------------
class HybridDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim,
                 layers=1, cell_type='LSTM', dropout=0.0, attention_enabled=True):
        super().__init__()
        self.use_attention = attention_enabled
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_input_dim = embed_dim + encoder_dim if attention_enabled else embed_dim
        output_input_dim = decoder_dim + embed_dim + (encoder_dim if attention_enabled else 0)

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(rnn_input_dim, decoder_dim, num_layers=layers,
                           dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.output_layer = nn.Linear(output_input_dim, vocab_size)

        if self.use_attention:
            self.attn = BahdanauAttention(encoder_dim, decoder_dim)

    def forward(self, input_token, hidden, enc_out, valid_positions):
        embedded = self.embedding(input_token).unsqueeze(1)

        if self.use_attention:
            last_hidden = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            attn_weights = self.attn(last_hidden, enc_out, valid_positions)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out)
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            attn_weights, context = None, None
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)
        flat_out = output.reshape(output.size(0), -1)
        flat_emb = embedded.reshape(embedded.size(0), -1)

        if self.use_attention:
            flat_ctx = context.reshape(context.size(0), -1)
            final_input = torch.cat((flat_out, flat_ctx, flat_emb), dim=1)
        else:
            final_input = torch.cat((flat_out, flat_emb), dim=1)

        logits = self.output_layer(final_input)
        return logits, hidden, attn_weights

# ------------------ Seq2Seq Wrapper ------------------
class SequenceTranslator(nn.Module):
    def __init__(self, encoder, decoder, pad_token_id, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.pad_id = pad_token_id
        
        self.decoder = decoder       
        self.out_dim = decoder.output_layer.out_features

    def forward(self, source, src_lengths, target):
        encoder_outputs, hidden = self.encoder(source, src_lengths)
        B, T = target.size()
        valid_positions = (source != self.pad_id)
        outputs = torch.zeros(B, T - 1, self.out_dim, device=self.device)

        token_sequence = zip(range(1, T), target[:, :-1].T, target[:, 1:].T)
        for t_idx, current_tok, next_tok in token_sequence:
            logits, hidden, _ = self.decoder(current_tok, hidden, encoder_outputs, valid_positions)
            outputs[:, t_idx - 1] = logits
        return outputs

    def infer_greedy(self, vocab, src_lens, src_seq, max_length=50   ):
        B = src_seq.size(0)

        current = torch.full((B,), vocab.get_sos_index(), dtype=torch.long, device=self.device)
        
        valid_positions = (src_seq != self.pad_id)
        enc_result, state = self.encoder(src_seq, src_lens)        
        
        all_preds = []

        for _ in range(max_length):
            logits, state, _ = self.decoder(current, state, enc_result, valid_positions)
            current = logits.argmax(1)
            all_preds.append(current.unsqueeze(1))

            if (current == vocab.get_eos_index()).all():
                break

        return torch.cat(all_preds, dim=1)
    


# ---------------------------------------------------------------------
# Legacy TensorFlow Model (for reference / was too slow so sihted to pytorch
# ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
class EncoderOptimized(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_size, hid_size,
                 layers=1, cell="LSTM", dropout=0.0):
        super().__init__()
        
        # Optimized embedding with better initialization
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, emb_size, mask_zero=True,
            embeddings_initializer='uniform')

        rnn_cls = dict(LSTM=tf.keras.layers.LSTM,
                       GRU=tf.keras.layers.GRU)[cell]

        # Pre-create RNN layers with optimizations
        self.rnn_layers = []
        for i in range(layers):
            layer = rnn_cls(
                hid_size,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                recurrent_dropout=0.0,  # Disable recurrent dropout for speed
                implementation=2,  # Use optimized implementation
                unroll=False  # Don't unroll for memory efficiency
            )
            self.rnn_layers.append(layer)

    @tf.function
    def call(self, src, training=False):
        """Optimized forward pass with TF function compilation."""
        x = self.embedding(src)
        mask = self.embedding.compute_mask(src)

        states = []
        for rnn in self.rnn_layers:
            x, *st = rnn(x, mask=mask, training=training)
            states.append(tuple(st))

        return x, states, mask

# ---------------------------------------------------------------------
class BahdanauAttentionOptimized(tf.keras.layers.Layer):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        # Use smaller intermediate dimension for speed
        self.attn = tf.keras.layers.Dense(
            dec_hid, use_bias=True, 
            kernel_initializer='glorot_uniform'
        )
        self.v = tf.keras.layers.Dense(
            1, use_bias=False,
            kernel_initializer='glorot_uniform'
        )

    @tf.function  # Compile for speed
    def call(self, dec_hidden, enc_outputs, src_mask):
        """Optimized attention computation."""
        batch_size = tf.shape(enc_outputs)[0]
        seq_len = tf.shape(enc_outputs)[1]
        
        # More efficient expansion
        dec_hidden_expanded = tf.broadcast_to(
            dec_hidden[:, tf.newaxis, :], 
            [batch_size, seq_len, tf.shape(dec_hidden)[1]]
        )

        # Concatenate and compute energy
        combined = tf.concat([dec_hidden_expanded, enc_outputs], axis=-1)
        energy = tf.nn.tanh(self.attn(combined))
        scores = tf.squeeze(self.v(energy), axis=-1)

        # Efficient masking with smaller negative value for numerical stability
        scores = tf.where(src_mask, scores, -1e4)
        
        # Compute attention weights and context
        attn_w = tf.nn.softmax(scores, axis=-1)
        context = tf.reduce_sum(
            enc_outputs * tf.expand_dims(attn_w, -1), 
            axis=1
        )
        
        return context, attn_w

class DecoderOptimized(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_size,
                 enc_hid, dec_hid,
                 layers=1, cell="LSTM", dropout=0.0, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        
        # Optimized embedding
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, emb_size,
            embeddings_initializer='uniform'
        )

        rnn_input_dim = emb_size + (enc_hid if use_attn else 0)
        rnn_cls = dict(LSTM=tf.keras.layers.LSTM,
                       GRU=tf.keras.layers.GRU)[cell]

        # Optimized RNN layers
        self.rnn_layers = []
        for i in range(layers):
            layer = rnn_cls(
                dec_hid,
                return_state=True,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=0.0,  # Disable for speed
                implementation=2,  # Optimized implementation
                unroll=False
            )
            self.rnn_layers.append(layer)

        # Attention layer
        self.attention = (BahdanauAttentionOptimized(enc_hid, dec_hid)
                          if use_attn else None)

        # Output layer with optimized initialization
        fc_in = dec_hid + emb_size + (enc_hid if use_attn else 0)
        self.fc = tf.keras.layers.Dense(
            vocab_size, 
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )

    @tf.function  # Compile the one_step function
    def one_step(self, inp_tok, prev_states, enc_out, src_mask, training=False):
        """Optimized single step decoding."""
        x = self.embedding(inp_tok)
        
        # Initialize context
        if self.use_attn:
            # Get last layer hidden state for attention
            last_state = prev_states[-1]
            dec_h = last_state if isinstance(last_state, tf.Tensor) else last_state[0]
            ctx, attn_w = self.attention(dec_h, enc_out, src_mask)
            x = tf.concat([x, ctx], axis=-1)
        else:
            ctx = tf.zeros_like(enc_out[:, 0, :])
            attn_w = None

        # Process through RNN layers
        states = []
        current_input = x[:, tf.newaxis, :]  # Add time dimension
        
        for rnn, prev_state in zip(self.rnn_layers, prev_states):
            output, *new_state = rnn(current_input, initial_state=prev_state, training=training)
            states.append(tuple(new_state))
            current_input = output[:, tf.newaxis, :]

        # Output is already [B, H], no need to squeeze
        final_output = output

        # Compute logits
        fc_input = tf.concat([
            final_output,
            self.embedding(inp_tok),
            ctx if self.use_attn else tf.zeros_like(final_output)
        ], axis=-1)
        
        logits = self.fc(fc_input)

        return logits, states, attn_w

# ---------------------------------------------------------------------
class Seq2SeqOptimized(tf.keras.Model):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def call(self, inputs, training=False):
        """Simplified forward pass without tf.function compilation."""
        src, src_len, tgt = inputs
        
        # Encode
        enc_out, dec_states, src_mask = self.encoder(src, training=training)

        # Use the original approach - unstack and iterate
        tokens = tf.unstack(tgt, axis=1)      # Python list of length Ttgt
        inp_tok = tokens[0]                   # <sos>
        logits_list = []

        for gold_tok in tokens[1:]:
            logit, dec_states, _ = self.decoder.one_step(
                inp_tok, dec_states, enc_out, src_mask, training=training)
            logits_list.append(logit)
            inp_tok = gold_tok                # teacher forcing

        return tf.stack(logits_list, axis=1)  # [B, Ttgt-1, V]

    @tf.function
    def infer_greedy(self, src, src_len, max_len=50):
        """Simplified greedy inference."""
        enc_out, dec_states, src_mask = self.encoder(src, training=False)
        inp_tok = tf.fill([tf.shape(src)[0]], self.sos_idx)
        outs = []

        for _ in tf.range(max_len):
            logit, dec_states, _ = self.decoder.one_step(
                inp_tok, dec_states, enc_out, src_mask, training=False)
            inp_tok = tf.argmax(logit, axis=-1, output_type=tf.int32)
            outs.append(inp_tok[:, tf.newaxis])
            if tf.reduce_all(tf.equal(inp_tok, self.eos_idx)):
                break

        return tf.concat(outs, axis=1)        # [B, <=max_len]

# Convenience functions to maintain compatibility
def Encoder(*args, **kwargs):
    return EncoderOptimized(*args, **kwargs)

def Decoder(*args, **kwargs):
    return DecoderOptimized(*args, **kwargs)

def Seq2Seq(*args, **kwargs):
    return Seq2SeqOptimized(*args, **kwargs)