from pymongo import MongoClient
from sklearn.preprocessing import normalize
import numpy as np
import os
import pandas as pd
import implicit
import scipy.sparse as sps

from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix

import logging
import sys
import csv
import math
import itertools
from itertools import combinations
import datetime
import time


mongo_ip = "127.0.0.1"
mongo_port = 27018

mc = MongoClient(host=mongo_ip, port=mongo_port,readPreference='primary') 
db = mc.prod_data

class Apriori():

	def __init__(self,client,version):
		self.client = client
		self.version = version
		self.all_prods = []

	def load_data(self):

		## loading last 100 days data from mongo baskets table
		print('Running function to load data from mongo baskets table')
		start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).replace(hour=0, microsecond=0, minute=0, second=0)
		end_date = datetime.datetime.now().replace(hour=0, microsecond=0, minute=0, second=0)

		data_cur = db.basket_products_data.find({'client':self.client,'date':{'$gte':start_date,'$lte':end_date}})

		data = pd.DataFrame(
		[
			{
				"date": item.get("date", ""),
				"products_list": item.get("products_list",[]),
				"action": item.get("action", ""),
			} for item in data_cur
		]
		)

		print('Number of basket entries loaded',len(data))

		## getting the list of all unique products
		unique_products = np.unique(list(itertools.chain.from_iterable(data['products_list'])))
		
		## error condition for WforWoman data
		if self.client==1067:
			unique_products=unique_products[unique_products!='w_']
			data['products_list'] = [[j for j in x if j!='w_'] for x in data['products_list']]
			data['len'] = [len(x) for x in data['products_list']]
			data = data[data['len']>1]

		print(data.head())

		print('Number of unique products in all data',len(unique_products))
		self.all_prods = unique_products
		self.data = data

	def matrix_creation(self):
		print('Running function to perform matrix creation')
		# number of unique products will be the dimension of the square matrix
		self.length = len(self.all_prods)

		## Initialize sparse matrix
		prod_matrix = dok_matrix((self.length, self.length), dtype=int)

		## creating matrix using data
		for d in range(len(self.data)):
			## create combinations of 2 for all products present within one basket
			combs = list(combinations(self.data['products_list'].iloc[d],2))
			# for each of these combinations, increase count in the square matrix for that product pair indices
			for c in combs:
				indx1 = np.where(self.all_prods == c[0])[0][0]
				indx2 = np.where(self.all_prods == c[1])[0][0]
				# indx1 = self.all_prods.index(c[0])
				# indx2 = self.all_prods.index(c[1])
				prod_matrix[indx1,indx2] += 1

		print('Shape of matrix',prod_matrix.shape)

		# input matrix created
		self.prod_matrix = csr_matrix(prod_matrix)

	def compute_metrics(self):
		## The below three resulting matrices that are summed along an axis result into a dense matrix
		print('Running function to perform computation of confidence and lift')
		# addition of matrix along row
		total_occurances  = self.prod_matrix.sum(1)
		# support for each product row-wise(probability of occurance along row)
		support_row_wise = self.prod_matrix.sum(1)/(self.prod_matrix.sum()/2)	
		# support for each product col-wise(probability of occurance along column)
		support_col_wise = self.prod_matrix.sum(0)/(self.prod_matrix.sum()/2)	

		print('Shape of total occurances',total_occurances.shape)
		print('Shape of row wise support',support_row_wise.shape)
		print('Shape of col wise support',support_col_wise.shape)

		## Scipy sparse matrix division to compute confidence
		## Confidence = input matrix / total occurance of each product
		val = np.repeat(total_occurances, self.prod_matrix.getnnz(axis=1)) 
		smoothing_param = 5
		confi_values = self.prod_matrix.data / (val + smoothing_param )
		c_ind = self.prod_matrix.indices
		c_indptr = self.prod_matrix.indptr
		confidence = csr_matrix( (np.ravel(confi_values),c_ind,c_indptr) )	

		print('Confidence computed',confidence.shape)
		print('Confidence computed',confidence.toarray())	
		# confidence = self.prod_matrix/(total_occurances+5) ## add 5 to smoothen the confidence of prods bought very few times
		# confidence[np.isnan(confidence)] = -1

		## Scipy sparse matrix division to compute lift
		## Lift = Confidence / Support (col-wise)
		val_l = np.repeat(support_col_wise, confidence.getnnz(axis=1)) 
		lift_values = confidence.data / val_l 
		l_ind = confidence.indices
		l_indptr = confidence.indptr
		lift = csr_matrix( (np.ravel(lift_values),l_ind,l_indptr) )	

		print('Lift computed',lift.shape)
		print('Lift computed',lift.toarray())		


		# lift = confidence/support_col_wise

		self.lift = lift

	def get_prod_pairs_and_insert(self):

		## sort indices of Lift matrix in desc order (best pair is max lift)
		print('Running function to sort and insert results into final mongo table')
		sorted_indices = np.argsort(-self.lift.toarray(),axis=1) ## do this 'X' rows at a time
		## determine how many top N products to keep for each product
			# max of either [100 or 1% of Total Unique Products ] 
		top_N = max((int(0.1*self.length)),100)

		## filter output matrix (sorted indices as per lift value) for topN
		if top_N<self.length:
			sorted_indices = sorted_indices[:,:top_N]
		else:
			top_N = sorted_indices.shape[0]

		## get actual product ids based on the top indices
		actual_prods = self.all_prods[sorted_indices]

		print('Results computed',actual_prods)

		## insert all top N ranks into output mongo table
		bulk = db.frequently_bought_together_products.initialize_unordered_bulk_op()

		counter=0

		updated_date = datetime.datetime.utcnow()
		st = time.time()
		for row_indx in range(len(self.all_prods)):

			for pair_indx in range(top_N):

				ori_product = self.all_prods[row_indx]
				pair_prod = actual_prods[row_indx,pair_indx]
				
				if ori_product!=pair_prod:
					bulk.find({'client':self.client,'productid':ori_product,'complementary_prod':pair_prod}).upsert().update(
						{'$set':{
								"client": self.client,
								"productid": ori_product,
								"complementary_prod": pair_prod,
								"rank": pair_indx,
								"version":self.version,
								"UpdatedOn":updated_date
								}}
						)
					counter=counter+1
					if (counter % 1000 == 0):
						print('data inserted',counter,'time',time.time()-st)
						st=time.time()
						bulk.execute()
						bulk = db.frequently_bought_together_products.initialize_ordered_bulk_op()
		if counter % 1000 != 0:
			bulk.execute()

		print('Mongo result insertion done')

		## Del all entries smaller then updated_date

		db.frequently_bought_together_products.delete_many({"UpdatedOn":{'$lt': updated_date}})

		print('Mongo older result deletion done')


	def Apriori_run(self):

		## loading all client's baskets data from mongo table
		self.load_data()
		## creating matrix from input data - prodsXprods matrix indicating the number of times both prods are ATCed or bought in same basket
		self.matrix_creation()
		## computing all Apriori metrics -> Confidence and Lift
		self.compute_metrics()
		## Insert final topN ranking-results for each product in output mongo table
		self.get_prod_pairs_and_insert()



