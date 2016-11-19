Best cases

ITEM CF K : 7, IGNORED = -1 -> 0.368
[7, 0, 'hamming'] -> 0.3845

USER CF K : 11, IGNORED = { 0 , -0.001 } -> 0.419
[11, -0.001, 'euclidean']

# No significant improvement
Case amplification: 1 produces the max
USER With INF: 0.418

# MF
Best K = 27, lb = 0.1, ignored = 0  -> ( 0.4827 )
Best K = 25, lb = 0.1, ignored = -0.1 -> ( 0.4821 )



# re-tune this
Best value of user range : 0.005,
ig = -0.1,
lb = 0.1,


##### Final stuff

Recommender 1:
# NMF
Best => K = 45, lb = 0.1 and ignored = -0.0005

# CF-QUERSTION
Recommender 2,3,4,5,6,7,8
K => 1,5,9,11,23,37,51
HAMMING Distance, Ig = 0

# CF-User
Recommender 9,10,11,12,13,14,15
K => 1,5,9,11,23,37,51
Euclidian Distance, Ig = -0.0005

# Tag - Rule
Recommender 16

# Word similarity
Recommender 17

# Char similarity
Recommender 18


# FEATURE REDUCTION
Question-Char : 434
Question-Word: 3265
User-Tag: 34
User-Char: 739
User-question: 2858


# GLAB
# MF { ranking_factorization_recommender }
IG: [0, -0.0001]

# WITH-CONTENT
# MF_CONTENT { ranking_factorization_recommender }
IG: [0, -0.0001]
33.5724677291, 33.6854086272
