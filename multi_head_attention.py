import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        if d_model % h != 0 :
            raise Exception("d_model must be a multiple of h")
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_mix = nn.Linear(d_model, d_model)

    def forward(self, x): #x.shape (batch_size, seq_len, d_model)
        Q = self.linear_q(x) # (batch_size, seq_len, d_model)
        # applique la transformation linéaire W^q @ embedding + b^q pour chacun des tokens
        Q = Q.view(Q.shape[0], Q.shape[1], self.h, self.d_k) # (batch_size, seq_len, h, d_k)
        Q = Q.transpose(1, 2) # (batch_size, h, seq_len, d_k)
        # On traite différentes dimensions par différentes têtes
        # L'objectif est de chercher différents types de patterns, et puisqu'on découpe 
        # les dimensions projetées de l'embedding (qui contient les fréquences du PE mixées), 
        # chaque tête peut potentiellement se spécialiser sur différentes échelles de distance
        # mais pas que, plein de patterns peuvent émerger.
        
        K = self.linear_k(x)
        K = K.view(K.shape[0], K.shape[1], self.h, self.d_k) # (batch_size, seq_len, h, d_k)
        K = K.transpose(1, 2)
        
        V = self.linear_v(x)
        V = V.view(V.shape[0], V.shape[1], self.h, self.d_k)
        V = V.transpose(1, 2)

        scores = Q @ K.transpose(-2,-1) # (batch_size, h, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-1)
        # pour chaque tête: distribution de probabilité d'attention entre positions

        output = scores @ V # (batch_size, h, seq_len, d_k)
        # pour chaque tête h: output[h, i] = somme pondérée des V[h, j] avec poids scores[h, i, j]
        output = output.transpose(1, 2) # (batch_size, seq_len, h, d_k)
        output = output.reshape(x.shape[0], x.shape[1], x.shape[2]) # (batch_size, seq_len, d_model)
        # reshape fusionne h et d_k et gère automatiquement la contiguïté (ce qui n'est pas le cas de view)

        output = self.linear_mix(output)
        # mixe les infos des têtes, c'est dans le papier Attention is all you Need

        return output # (batch_size, seq_len, d_model)
        # intuition : output[i] = somme pondérée des V[j] avec poids scores[i,j]