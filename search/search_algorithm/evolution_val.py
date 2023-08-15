# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import numpy as np
from tqdm import tqdm
import sys

__all__ = ['EvolutionFinder_val']


class ArchManager:
	def __init__(self):
		self.num_blocks =18#4464
		self.num_stages = 4

		#self.kernel_sizes = [3, 5, 7]
		self.expand_ratios = [0.2,0.25,0.35]
		self.depths = [0,1,2]
		self.resolutions = [24,32,40,48]
		self.width_mult=[0,1,2]
		#self.width_mult=[0.65,0.8,1.0]

	def random_sample_arch(self):
		sample = {}
		d = []
		e = []
		w=[]
		#ks = []
		for i in range(self.num_stages+1):
			d.append(random.choice(self.depths))

		for i in range(self.num_stages+2):
			w.append(random.choice(self.width_mult))


		for i in range(self.num_blocks):
			e.append(random.choice(self.expand_ratios))

		sample = {
			'w': w,
			'e': e,
			'd': d,
			'r': random.choice(self.resolutions)
		}

		return sample



	def mutate_resolution(self, arch_dict, mutate_prob):
		if random.random() < mutate_prob:
			arch_dict['r']=random.choice(self.resolutions)
		return arch_dict

	def mutate_arch(self, arch_dict, mutate_prob):
		for i in range(self.num_blocks):
			if random.random() < mutate_prob:
				arch_dict['e'][i] = random.choice(self.expand_ratios)

		for i in range(self.num_stages+1):
			if random.random() < mutate_prob:
				arch_dict['d'][i] = random.choice(self.depths)

		for i in range(self.num_stages+2):
			if random.random() < mutate_prob:
				arch_dict['w'][i] = random.choice(self.width_mult)

		return arch_dict


class EvolutionFinder_val:

	def __init__(self, efficiency_predictor,test_func,supernet,testLoader, **kwargs):
		self.efficiency_predictor = efficiency_predictor
		self.arch_manager=ArchManager()


		self.testLoader=testLoader
		self.supernet=supernet
		self.test_func=test_func

	

		# evolution hyper-parameters
		self.arch_mutate_prob = kwargs.get('arch_mutate_prob', 0.1)
		self.resolution_mutate_prob = kwargs.get('resolution_mutate_prob', 0.5)
		self.population_size = kwargs.get('population_size', 100)
		self.max_time_budget = kwargs.get('max_time_budget', 500)
		self.parent_ratio = kwargs.get('parent_ratio', 0.25)
		self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

	

	def update_hyper_params(self, new_param_dict):
		self.__dict__.update(new_param_dict)

	def random_valid_sample(self, constraint):
		while True:
			sample = self.arch_manager.random_sample_arch()
			efficiency = self.efficiency_predictor.get_efficiency(sample)
			#efficiency = self.efficiency_predictor.count_flops_given_config(sample)
			if efficiency <= constraint:
				return sample, efficiency

	def mutate_sample(self, sample, constraint):
		while True:
			new_sample = copy.deepcopy(sample)

			self.arch_manager.mutate_resolution(new_sample, self.resolution_mutate_prob)
			self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob)

			efficiency = self.efficiency_predictor.get_efficiency(new_sample)
			if efficiency <= constraint:
				return new_sample, efficiency

	def crossover_sample(self, sample1, sample2, constraint):
		while True:
			new_sample = copy.deepcopy(sample1)
			for key in new_sample.keys():
				if not isinstance(new_sample[key], list):
					new_sample[key] = random.choice([sample1[key], sample2[key]])
				else:
					for i in range(len(new_sample[key])):
						new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])

			efficiency = self.efficiency_predictor.get_efficiency(new_sample)
			if efficiency <= constraint:
				return new_sample, efficiency

	def run_evolution_search(self, constraint, verbose=False, **kwargs):
		"""Run a single roll-out of regularized evolution to a fixed time budget."""
		self.update_hyper_params(kwargs)

		mutation_numbers = int(round(self.mutation_ratio * self.population_size))
		parents_size = int(round(self.parent_ratio * self.population_size))

		best_valids = [-100]
		population = []  # (validation, sample, latency) tuples
		child_pool = []
		efficiency_pool = []
		best_info = None
		if verbose:
			print('Generate random population...')
		for _ in range(self.population_size):
			sample, efficiency = self.random_valid_sample(constraint)
			child_pool.append(sample)
			efficiency_pool.append(efficiency)

		accs=[]

		for arch in child_pool:
			acc=self.test_func(self.supernet,self.testLoader,arch)
			accs.append(acc)
		
		for i in range(self.population_size):
			population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

		if verbose:
			print('Start Evolution...')
		# After the population is seeded, proceed with evolving the population.
		with tqdm(total=self.max_time_budget, desc='Searching with constraint (%s)' % constraint,
		          disable=(not verbose)) as t:
			for i in range(self.max_time_budget):
				parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
				acc = parents[0][0]
				t.set_postfix({
					'acc': parents[0][0]
				})
				if not verbose and (i + 1) % 100 == 0:
					print('Iter: {} Acc: {}'.format(i + 1, parents[0][0]))

				if acc > best_valids[-1]:
					best_valids.append(acc)
					best_info = parents[0]
				else:
					best_valids.append(best_valids[-1])

				population = parents
				child_pool = []
				efficiency_pool = []

				for j in range(mutation_numbers):
					par_sample = population[np.random.randint(parents_size)][1]
					# Mutate
					new_sample, efficiency = self.mutate_sample(par_sample, constraint)
					child_pool.append(new_sample)
					efficiency_pool.append(efficiency)

				for j in range(self.population_size - mutation_numbers):
					par_sample1 = population[np.random.randint(parents_size)][1]
					par_sample2 = population[np.random.randint(parents_size)][1]
					# Crossover
					new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2, constraint)
					child_pool.append(new_sample)
					efficiency_pool.append(efficiency)
				
				#print(child_pool)
				#sys.exit()

				accs=[]

				for arch in child_pool:
					acc=self.test_func(self.supernet,self.testLoader,arch)
					accs.append(acc)
		
			
				for j in range(self.population_size):
					population.append((accs[j].item(), child_pool[j], efficiency_pool[j]))

				t.update(1)

		return best_valids, best_info
