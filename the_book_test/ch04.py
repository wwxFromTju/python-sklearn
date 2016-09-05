#!/usr/bin/env python
# encoding=utf-8

from collections import defaultdict
import sys
from operator import itemgetter

import pandas as pd

all_raings = pd.read_csv('data/ml-100k/u.data', delimiter='\t', header=None, names=['UserID', 'MovieID', 'Rating', 'Datetime'])
all_raings['Datetime'] = pd.to_datetime(all_raings['Datetime'], unit='s')

print(all_raings[all_raings['UserID'] == 675].sort_values('MovieID'), '\n')

all_raings['Favorable'] = all_raings["Rating"] > 3

print(all_raings[all_raings['UserID'] == 1][:5], '\n')

ratings = all_raings[all_raings['UserID'].isin(range(200))]

favorable_ratings = ratings[ratings['Favorable']]
print(favorable_ratings[:5], '\n')

favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby('UserID')['MovieID'])
print(len(favorable_reviews_by_users), '\n')

num_favorable_by_movie = ratings[['MovieID', 'Favorable']].groupby('MovieID').sum()
print(num_favorable_by_movie.sort_values('Favorable', ascending=False)[:5], '\n')


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])


frequent_itemsets = {}
min_support = 50

frequent_itemsets[1] = dict((frozenset((movie_id,)), row['Favorable'])
                            for movie_id, row in num_favorable_by_movie.iterrows()
                            if row['Favorable'] > min_support)
print("There are {} movies with more than {} favorable reviews.\n".format(len(frequent_itemsets[1]), min_support))
sys.stdout.flush()
for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1],
                                                   min_support)

    if len(cur_frequent_itemsets) == 0:
        print('Did not find any frequent itemsets of length {}'.format(k))
        sys.stdout.flush()
        break
    else:
        print('I found {} frequent itemsets od length {}'.format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets

del frequent_itemsets[1]

print('\nFound a total of {0} frequent itemsets\n'.format(sum(len(itemsets) for itemsets in frequent_itemsets.values())))

candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print('There are {} candidate rules\n'.format(len(candidate_rules)))

print(candidate_rules[:5], '\n')

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
                   for candidate_rule in candidate_rules}

min_confidence = 0.9

rule_confidence = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > min_confidence}
print(len(rule_confidence), 'n')

sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print('Rule #{}'.format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print('Rule: If a persion recommends {} they will also recommend {}'.format(premise, conclusion))
    print(' - Confidence: {:.3f}\n'.format(rule_confidence[(premise, conclusion)]))

movie_name_data = pd.read_csv('data/ml-100k/u.item', delimiter="|", header=None, encoding = "mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action", "Adventure",
                           "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data['MovieID'] == movie_id]['Title']
    title = title_object.values[0]
    return title

print(get_movie_name(4), '\n')

for index in range(5):
    print('Rule #{}'.format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print('Rule: If a person recommends {} they will also recommend {}'.format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print('')

test_dataset = all_raings[~all_raings['UserID'].isin(range(200))]
test_favorable = test_dataset[test_dataset['Favorable']]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby('UserID')["MovieID"])

print(test_dataset[:5])

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

test_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
                   for candidate_rule in rule_confidence}
print(len(test_confidence))

sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)
print(sorted_test_confidence[:5])

for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1)))
    print(" - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1)))
    print("")

