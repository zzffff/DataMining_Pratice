#!/usr/bin/env python
# coding: utf-8

# In[19]:


class Employee:
   """所有员工的基类"""
   empCount = 0
 
   def __init__(self, a, b):
      self.name = a
      self.salary = b
      Employee.empCount += 1
   
   def displayCount(self):
     print ("Total Employee %d" % Employee.empCount)
 
   def displayEmployee(self):
      print ("Name : ", self.name,  ", Salary: ", self.salary)


# In[20]:


e1 = Employee('Tom',5000) #self其实就是e1,在使用Employee这个class赋值给一个变量叫做实例化
                          #实例化的时候不用输入这个参数，只要输入后面的参数
e1.name


# In[22]:


e1.displayCount() #调用类里面的方法，直接用实例e1调用，然后在括号里传入除了self之外的参数（因为self也就是e1），就能得到结果


# In[24]:


e1.displayEmployee()

