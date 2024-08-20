#!/usr/bin/env python
# coding: utf-8

# In[1]:


def sortColors(self, nums):
    red, white, blue = 0, 0, len(nums)-1
    
    while white <= blue:
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            white += 1
            red += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1
    return nums
def finalArray(b):
    for k in b:
        print(k, end=' ')
num = input()
list1 = num.split()
for i in range(0, len(list1)):
    list1[i] = int(list1[i])
finallist = list(list1)
size = len(finallist)
final = sortColors(size,finallist)
finalArray(final)


# In[2]:


my_str = input()  

def ipvalid(ip): 
   ip = ip.split(".") 
      
   for i in ip: 
      if len(i) > 3 or int(i) < 0 or int(i) > 255: 
         return False
      if len(i) > 1 and int(i) == 0: 
         return False
      if len(i) > 1 and int(i) != 0 and i[0] == '0': 
         return False
   return True

def construct_dot(s, t):

    if t==0: return [s]
    new_list = []

    for p in range(1,len(s) - t + 1):

        new_str = str(s[:p]) + '.'
        res_str = str(s[p:]) 

        sub_list = construct_dot(res_str, t-1) 

        for sl in sub_list:
            new_list.append(new_str + sl)
    return new_list

all_list = []
for n_dots in range(len(my_str)):
    all_list.extend(construct_dot(my_str,3))

def filter_1(list):
    final_list =[]
    for i in list:
            if ipvalid(i) == True:
                final_list.append(i)
    final_list1 =set(final_list)
    return(final_list1)
print(' '.join(map(str,filter_1(all_list) )))


# In[ ]:




