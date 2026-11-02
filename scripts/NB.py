from scripts.count import total_words
from scripts.bow import clean
from math import log

'''
Notes: Values initially zeroed out immediately because values got very very close to zero.
To prevent this, we summed logs instead of repeatedly multiplying. Learned this from

https://web.stanford.edu/~jurafsky/slp3/B.pdf
'''

class NB_for_2:
    def __init__(s,A,B,k):
        print("\nCreating NB Classifier...")
        s.a, s.b = total_words(A), total_words(B)

        aub = s.a + s.b
        
        s.dictionary = set(A.keys()) | set(B.keys())

        s.A = A
        s.B = B
        s.k = k

        # probablity a word is from A, B
        s.P_A = s.a/aub
        s.P_B = s.b/aub

        print("NB Classifier created")

    def P_word_given_A(s,w):
        return (s.A.get(w,0) + s.k)/(s.a + s.k*len(s.dictionary))

    def P_word_given_B(s,w):
        return (s.B.get(w,0) + s.k)/(s.b + s.k*len(s.dictionary))

    # def P_message_given_A(s,p): #p is cleaned
    #     P = 1
    #     for w in p:
    #         P*= s.P_word_given_A(w)
    #     return P

    # def P_message_given_B(s,p): #p is cleaned
    #     P = 1
    #     for w in p:
    #         P*= s.P_word_given_B(w)
    #     return P
    
    # def P_message(s,p):
    #     return s.P_message_given_A(p)*s.P_A + s.P_message_given_B(p)*s.P_B

    def log_score_A(s,p): #p is cleaned
        #the numerator in P(A|message)
        return log(s.P_A)+sum([log(s.P_word_given_A(w)) for w in p])
    
    def log_score_B(s,p): 
        return log(s.P_B)+sum([log(s.P_word_given_B(w)) for w in p])

    # def P_A_given_message(s,p):
    #     denom = s.P_message(p)
    #     if denom==0:
    #         return 0.5
    #     return s.P_message_given_A(p)*s.P_A/denom
    
    def classify_paragraph(s,p):
        '''
        returns
        0 : A
        1 : B
        '''
        #clean
        p = [x for x in map(clean,p.split()) if x!=None and x in s.dictionary]
        return 0 if s.log_score_A(p)>=s.log_score_B(p) else 1 


