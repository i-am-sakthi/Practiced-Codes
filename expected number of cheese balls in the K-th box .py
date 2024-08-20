#!/usr/bin/env python
# coding: utf-8

# In[1]:


def expected_cheese_balls(T, test_cases):
    results = []
    for i in range(T):
        B, M, N, K = test_cases[i]

        expected_balls = [0] * (B + 1)
        expected_balls[1] = M

        # Iterate through each box
        for j in range(1, B):
            probability = 1 / (2 ** j)

            for k in range(1, j + 1):
                expected_balls[j + 1] += probability * expected_balls[j - k + 1]


        results.append(expected_balls[K])

    return results

def main():
    T = int(input().strip())
    test_cases = []

    for _ in range(T):
        B, M, N, K = map(int, input().strip().split())
        test_cases.append((B, M, N, K))

    results = expected_cheese_balls(T, test_cases)

    for result in results:
        print("{:.10f}".format(result))

if __name__ == "__main__":
    main()


# In[ ]:


2
3 4 4 2
1 3 2 1

