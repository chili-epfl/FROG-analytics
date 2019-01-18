from analytics import operation_builder
from analytics.Operations import ElementaryOperation, Paragraph, SuperParagraph, Operation
import config
from itertools import combinations
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy import optimize
from scipy import spatial
import spacy
import regex as re
from spacy.tokenizer import Tokenizer
import warnings




class WindowOperation:
    '''
    create a new operation based on windows
    '''
    def __init__(self,groupNum,author,ops,time_interval=1000000):
        '''
        :param groupNum: the group that window belongs to
        :param author: the author of this window
        :param ops: the operations in the window
        :param time_interval: time interval of the window
        '''
        self.groupNum = groupNum
        self.author = author
        self.operations = ops
        self.elemOps = []
        self.start_time =  float('inf')
        self.time_interval = time_interval
        self.text = ''
        self.textVector=[]
        self.endTime = 0.0
        self.text_added_len = 0


    def addOperation(self,op):
        self.operations.append(op)

    def generateElemOps(self):
        '''
        used to sort the elem_operation and define the end time of the window
        :return: None
        '''
        for op in self.operations:
            if self.endTime<op.timestamp_end:
                self.endTime= op.timestamp_end
            if self.start_time>op.timestamp_start:
                self.start_time = op.timestamp_start
            self.elemOps.extend(op.elem_ops)
        self.elemOps = ElementaryOperation.sort_elem_ops(self.elemOps)


    def __eq__(self, other):

        return (self.author==other.author) and (self.groupNum==other.groupNum)


    def createWindowText(self,vector_flag,model):
        '''
        :param vector_flag:  whether to compute vector
        :param model:  pre-trained model
        :return:
        '''
        text = ''
        for op in self.operations:
            op.getOpText()
            if op.text != '\n':
                text +=  op.text + ' '  # compute the length of text added
        self.text =  text
        #a = text.split()
        self.text_added_len = len(text.split())
        if vector_flag:
            cleaned_text = cleanText(text)
            self.textVector = model.embed_sentence(cleaned_text)
