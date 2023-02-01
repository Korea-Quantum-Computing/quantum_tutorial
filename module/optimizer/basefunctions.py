import numpy as np

def get_bin(x, p):
    '''
        선택된 변수의 정수 index를 [01100110..] 방식으로 변환해주는 함수
        input: 정수 index array, 총 변수 개수 p
        output: binary 방식 변수 선택 결과
    '''
    zeros = np.zeros(p, dtype=int)
    zeros[x] = 1
    return zeros

def get_index(theta):
    return(np.where(theta)[0])

def flip(k, x, p):
    '''
        기존 선택된 변수들에서 k개만큼 flip해주는 함수
        input: flip할 횟수 k, 정수 index array, 총 변수 개수 p
        output: 새롭게 선택된 변수 결과
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(p, size = k, replace = False)
    zeros[idx] = 1
    new = abs(x - zeros)
    return new

def flip2(k, x, p):
    '''
    k : 몇 개 뒤집을 것인지, 2의 배수여야 함
    x : 뒤집을 대상
    p : 총 변수 개수
    '''
    x_array = np.asarray(x)
    one = get_index(x_array==1)
    zero = get_index(x_array==0)
    idx_onetozero = np.random.choice(one, size = int(k/2), replace = False).tolist()
    idx_zerotoone = np.random.choice(zero, size = int(k/2), replace = False).tolist()
    x_array[idx_onetozero] = 0
    x_array[idx_zerotoone] = 1
    return(x_array.tolist())

def get_QB(theta_temp,Q, beta, lmbd):
    Q = np.asarray(Q)
    theta_temp = np.asarray(theta_temp)
    beta = np.asarray(beta)
    return lmbd * theta_temp.T @ Q @ theta_temp + (1 - lmbd) * beta.T @ theta_temp
