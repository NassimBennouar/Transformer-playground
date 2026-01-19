Pourquoi le positional encoding ?

Contrairement aux RNNs qui traitent les mots d'une phrase séquentiellement et qui prennent intrinséquement en compte leur ordre, les transformers se basent sur un mécanisme de multi-head self-attention.

Ce mécanisme accélère grandement le temps d'entraînement des modèles de langage et peuvent théoriquement prendre en compte des dépendances plus longues entre deux mots.

Une telle accélération est possible parce que plusieurs mots d'une même phrase peuvent passer en même temps dans la stack encoder/decoder du Transformer, ce qui nécessite alors une manière d'incorporer l'ordre des mots dans le modèle.

Dans l’architecture originale du Transformer, l’information de position est injectée en amont du modèle en ajoutant un positional encoding aux embeddings des tokens.

Par la suite, de nombreuses extensions ont été proposées pour dépasser les limites de cette approche, notamment des encodages positionnels relatifs comme RoPE, qui modifient directement le calcul de l’attention.

Nous n'allons cependant pas en parler en cela qu'on étudie l'architecture initiale, et que la méthode employée par les chercheurs permet de comprendre le problème de façon assez simple.

Intuitivement, on peut avoir des idées sur comment développer ces encodages positionnels :

*Peut-être affecter un nombre dans [0, 1] pour chaque mot d'une séquence, 0 au premier mot, 1 au dernier mot, 0 + 1 / N pour les autres avec N la longueur de la séquence ?*

Le problème serait que le modèle ne pourrait pas savoir le nombre de mots présents dans cette intervalle, et donc ne pourrait pas évaluer les distances

*Peut-être affecter une numérotation 0 1 2 3... en fonction de la position ?*

Le premier problème ici est la généralisation du modèle, comment va-t-il réagir face à des séquences d'une longueur qu'il n'a jamais vu pendant l'entraînement ? Le deuxième problème, c'est que la numérotation brute rend difficile au modèle la représentation des décalages, bien que pour nous autres -- simples mortels -- c'est la représentation la plus intuitive.

Il nous faut donc une solution qui respecte ces différentes caractéristiques :

- Elle doit donner un encodage unique pour chaque position
- La distance entre deux position doit être consistante peu importe la longueur de la séquence
- Elle doit permettre au modèle de généraliser à des longueurs plus grandes sans effort
- Elle doit être déterministe (afin que le modèle apprenne des modèles stables et qu'à l'inférence une même séquence en entrée soit la même)

Il se trouve que la solution proposée par les chercheurs respecte ces caractéristiques, décrivons la.

Les séquences passées sont d'abord embeddées, donnant une matrice pour chaque séquence de $(seq len, d)$ avec $d$ la dimension de l'embedding et $seq len$ le nombre de tokens.

L'information de positionnement est simplement ajoutée à la valeur de l'embedding selon la formule suivante

Soit $t$ la position dans la séquence, $\vec{p}_t \in \mathbb{R}^{d}$ son encodage positionnel (qui sera additionné à l'embedding), et $d$ la dimension de l'embedding

On a $f:\mathbb{N}\longrightarrow \mathbb{R}^{d}$ la fonction qui produit le vecteur $\vec{p}_t$ :

$$\vec{p}_t^{(i)} = f(t)(i) :=
\begin{cases}
\sin(\omega_k t), & \text{si } i = 2k \\
\cos(\omega_k t), & \text{si } i = 2k + 1
\end{cases}$$

où l'angle $\omega_k$ est

$$\omega_k = \frac{1}{10000^{2k/d}}$$

duquel on déduit la période $\lambda_k$

$$\lambda_k=2\pi \cdot 10000^{2k/d}$$

On observe qu'on a $d/2$ pairs cosinus/sinus, et que la fréquence diminue le long de la dimension du vecteur ($k$ allant de $0$ à $d/2$). La période va de $2\pi$ pour la première dimension à $2\pi \cdot 10000$ pour la dernière.

$$\vec{p}_t = \begin{bmatrix}
\sin(\omega_1 \cdot t) \\
\cos(\omega_1 \cdot t) \\
\\
\sin(\omega_2 \cdot t) \\
\cos(\omega_2 \cdot t) \\
\\
\vdots \\
\\
\sin(\omega_{d/2} \cdot t) \\
\cos(\omega_{d/2} \cdot t)
\end{bmatrix} \in \mathbb{R}^{d\times1}$$

Bref, si les esprits les plus mathématiques ont déjà compris pourquoi ça marche, les simples informaticiens desquels je fais partie on besoin d'un peu plus d'explication.

Je vais donc prendre un exemple qui nous parle, représentons des nombres au format binaire

<table>
  <tr>
    <th>Déc</th><th>Bits</th>
    <th>Déc</th><th>Bits</th>
  </tr>
  <tr>
    <td>0</td><td><span style="color:red">0 </span><span style="color:green">0 </span><span style="color:blue">0 </span><span style="color:orange">0 </span></td>
    <td>8</td><td><span style="color:red">1 </span><span style="color:green">0 </span><span style="color:blue">0 </span><span style="color:orange">0 </span></td>
  </tr>
  <tr>
    <td>1</td><td><span style="color:red">0 </span><span style="color:green">0 </span><span style="color:blue">0 </span><span style="color:orange">1 </span></td>
    <td>9</td><td><span style="color:red">1 </span><span style="color:green">0 </span><span style="color:blue">0 </span><span style="color:orange">1 </span></td>
  </tr>
  <tr>
    <td>2</td><td><span style="color:red">0 </span><span style="color:green">0 </span><span style="color:blue">1 </span><span style="color:orange">0 </span></td>
    <td>10</td><td><span style="color:red">1 </span><span style="color:green">0 </span><span style="color:blue">1 </span><span style="color:orange">0 </span></td>
  </tr>
  <tr>
    <td>3</td><td><span style="color:red">0 </span><span style="color:green">0 </span><span style="color:blue">1 </span><span style="color:orange">1 </span></td>
    <td>11</td><td><span style="color:red">1 </span><span style="color:green">0 </span><span style="color:blue">1 </span><span style="color:orange">1 </span></td>
  </tr>
  <tr>
    <td>4</td><td><span style="color:red">0 </span><span style="color:green">1 </span><span style="color:blue">0 </span><span style="color:orange">0 </span></td>
    <td>12</td><td><span style="color:red">1 </span><span style="color:green">1 </span><span style="color:blue">0 </span><span style="color:orange">0 </span></td>
  </tr>
  <tr>
    <td>5</td><td><span style="color:red">0 </span><span style="color:green">1 </span><span style="color:blue">0 </span><span style="color:orange">1 </span></td>
    <td>13</td><td><span style="color:red">1 </span><span style="color:green">1 </span><span style="color:blue">0 </span><span style="color:orange">1 </span></td>
  </tr>
  <tr>
    <td>6</td><td><span style="color:red">0 </span><span style="color:green">1 </span><span style="color:blue">1 </span><span style="color:orange">0 </span></td>
    <td>14</td><td><span style="color:red">1 </span><span style="color:green">1 </span><span style="color:blue">1 </span><span style="color:orange">0 </span></td>
  </tr>
  <tr>
    <td>7</td><td><span style="color:red">0 </span><span style="color:green">1 </span><span style="color:blue">1 </span><span style="color:orange">1 </span></td>
    <td>15</td><td><span style="color:red">1 </span><span style="color:green">1 </span><span style="color:blue">1 </span><span style="color:orange">1 </span></td>
  </tr>
</table>

On observe différents taux de changement entre les bits. Le LSB change à chaque nombre, le deuxième moins signifiant deux fois moins, et ainsi de suite.

Dans notre exemple, si on passe de la position 0 (0000) à la position 8 (1000), le LSB aura changé de valeur 8 fois, contre une seule fois pour le MSB.

C'est, en version discrète, ce qu'il se passe avec le positional encoder : la première paire cosinus/sinus a la fréquence maximale possible, $\omega_0 = 1$, il faut alors environ 6 à 7 positions pour compléter un cycle de $2\pi$ radians (un "tour de cercle" complet du point $(\sin(\omega_0 t), \cos(\omega_0 t))$ ). Cette dimension oscille donc rapidement, capturant des patterns locaux, à l'image du LSB qui change fréquemment.

À l'inverse, pour les fréquences minimales où $\omega_k \approx 1/10000$ (quand $k$ est proche de $d/2$), il faut environ $2\pi \times 10000 \approx 62832$ positions pour compléter un cycle. Ces dimensions oscillent très lentement, capturant des patterns globaux sur de longues distances, à l'image du MSB qui ne change que rarement.

C'est cette variété de fréquences d'oscillation, allant du local au global, que l'on introduit en amont de l'encodeur et qui permet au modèle de capturer différents patterns de dépendances selon leur échelle.

Ce que nous avons dit permet déjà de valider deux de notre quatre point :
- La fonction est bien déterministe
- Elle donne un encodage unique pour chaque position car croissante

Mais si nous voulons valider les deux derniers points qui sont :

- qu'elle permet au modèle de généraliser à des longueurs plus grandes sans effort

- que la distance entre deux positions est consistante peu importe la longueur de la séquence

Alors nous devons aller un peu plus loin et prouver qu'on a, comme décrit dans le papier, pour tout offset fixe $\phi$ , $\vec{p}\_{t+\phi}$ qui peut être représenté comme une fonction linéaire de $\vec{p}_{t}$.

Ca va être très rapide

Posons cette transformation linéaire $M \in \mathbb{R}^{2 \times 2}$ :

$$M \cdot
\begin{bmatrix}
\sin(\omega_k \cdot t) \\
\cos(\omega_k \cdot t)
\end{bmatrix} =
\begin{bmatrix}
\sin(\omega_k \cdot (t+\phi)) \\
\cos(\omega_k \cdot (t+\phi))
\end{bmatrix}$$

Disons

$$
M = \begin{bmatrix}
m & n \\
o & p
\end{bmatrix}
$$

Sachant que

$$
\begin{aligned}
\cos(a+b) &= \cos a \cdot \cos b - \sin a \cdot \sin b \\
\sin(a+b) &= \sin a \cdot \cos b + \cos a \cdot \sin b
\end{aligned}
$$

Alors

$$\begin{bmatrix} m & n \\
o & p \end{bmatrix} \cdot \begin{bmatrix} \sin(\omega_k \cdot t) \\
\cos(\omega_k \cdot t) \end{bmatrix} = \begin{bmatrix} sin(\omega_k \cdot t) \cdot cos(\omega_k \cdot \phi) +\ cos(\omega_k \cdot t) \cdot sin(\omega_k \cdot \phi) \\
cos(\omega_k \cdot t) \cdot cos(\omega_k \cdot \phi) \ -\ sin(\omega_k \cdot t) \cdot sin(\omega_k \cdot \phi) \end{bmatrix}$$


On déduit

$$M = \begin{bmatrix}\cos(\omega_k \phi) & \sin(\omega_k \phi) \\
-\sin(\omega_k \phi) & \cos(\omega_k \phi) \end{bmatrix}$$

Et voilà !

Pour résumer, ce que vous devez retenir pour construire votre intuition sur cet élément, c'est :

1. Le positional encoding injecte une structure qui rend les décalages linéarisables, de manière à ce que l'information de position relative soit exploitable par le modèle

2. De nombreuses fréquences sont fournies, ce qui permet de capturer des patterns plus ou moins locaux (imaginez une horloge qui a une trotteuse, une aiguille pour les minutes et une autre pour les heures)

Je vous recommande d'implémenter cette brique avec PyTorch sans autocomplete et sans recopier (dans la mesure de votre connaissance de PyTorch), à partir de ce que vous avez compris.
