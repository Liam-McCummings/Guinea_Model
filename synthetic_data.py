import random
import math

input_vector = (N, E, sigx, sigy, sigz)
m_e = 9.1093837e-31
c = 2.99792458e8
output = 0 

def seperate_vector(input_vector):
    N = input_vector[0]
    E = input_vector[1]
    sigx = input_vector[2]
    sigy = input_vector[3]
    sigz = input_vector[4]
    return N, E, sigx, sigy, sigz

def get_lumi(input_vector):
    N, E, sigx, sigy, sigz = seperate_vector(input_vector)
    geometric_lumi = N**2/(4*math.pi*sigx*sigy)
    #pinching causes lumi to increase as the beams interact, guineapig simulates this pinching out but for efficiency we are just
    #going to assume the factor is proportional to the disruption factor

    gamma = E/(m_e*c**2)
    disruption_factor = (2*N*r_e*sigz)/(gamma*sigy*(sigx+sigy))
    

    def f(x):
        1 + A*tanh()
    #logarithmic correlation for the new lumi makes sense since the lumi will flatten out since can only be so pinched, and 
    #will increase quickly with initial pinching
    effective_lumi = math.log(disruption_factor)*geometric_lumi
    return effective_lumi
