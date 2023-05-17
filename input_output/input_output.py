import pandas as pd
import csv
import math
import pdb
import numpy as np
import time
from numba import jit
import copy


@jit
def reshape_output(output):
    #transform output (shape # molecules, # assays, # data) into (# molecules, # assays x # data)
    processed_output = np.zeros((507, 229432), dtype=np.float32) 
    for i in range(507): # for each uneven 2d array in output
        counter = 0
        for j in range(len(output[i])): #convert into 1d array
            for k in range(len(output[i][j])):
                processed_output[i][counter] = float(output[i][j][k])
                counter += 1
    return processed_output

#@jit
def get_input_output():
    # #how many of the ids are in every assay
    print("Assembling toxid to cid hash")
    cid_to_toxid = {}
    id_pairs_file = open("all_id_pairs.txt")
    whole_thing = id_pairs_file.read()
    id_pairs = whole_thing.split('\n')
    for pair in id_pairs:
        split = pair.split("\t")
        toxid, cid = split[0], split[1]
        cid_to_toxid[cid] = toxid
    id_pairs_file.close()

    # #all important output cids guaranteed to be in input toxids
    # #just have to make sure we get rid of all input toxids that aren't in output cids

    # #PROCESS EACH PART OF OUTPUT AS IT GOES

    print("Assembling output")
    print("part", 1)
    load_file = open("output_npy/output_part"+str(1)+".npy", "rb")
    output = reshape_output(np.load(load_file, allow_pickle=True))
    load_file.close()
    for i in range(2, 11):
        print("part", i)
        load_file = open("output_npy/output_part"+str(i)+".npy", "rb")
        output = np.concatenate((output, reshape_output(np.load(load_file, allow_pickle=True))))
        load_file.close()


    #MAKE SURE INPUT ASSEMBLED IN SAME CID ORDER AS OUTPUT 

    print("Assembling input")
    tox21_input = open("tox21_dense_train.csv")
    input_csv = csv.reader(tox21_input)
    toxid_to_input = {} 
    for row in input_csv:
        if (len(row[0]) > 0 and type(row[0]) == str):
            toxid_to_input[row[0]] = np.array(row[1:], dtype=np.float32)
    tox21_input.close()

    input = np.zeros((5076, 801), dtype=np.float32)
    load_file = open("output_npy/cid_in_order.npy", "rb")
    cids = np.load(load_file, allow_pickle=True)
    load_file.close()
    counter = 0
    f = open("cids_in_order.txt", "a")
    for cid in cids:
        cid = str(int(cid))
        if cid_to_toxid[cid] in toxid_to_input:
            #print(cid)
            f.write(cid + "\n")
            input[counter] = toxid_to_input[cid_to_toxid[cid]]
            counter += 1
        else:
            print("uh oh...")

    input = input[:5070] #somehow the last 6 cids got cut off from output, TODO fix this issue (prob to do with splitting into files)
    
    f.close()
    pdb.set_trace()

    #want cid at index 4058

    print("len input:", len(input))
    print("len output:", len(output))

    np.save("input.npy", input)
    np.save("output.npy", output)

    # np.save("short-input.npy", input[:20])
    # np.save("short-output.npy", output[:20])

    return input, output



get_input_output()



"""" --------------------------------------------------------------------------------------------------------------------------------"""
#what i did to go from assays to output txt files
# for name in assay_names:
#     count += 1
#     print("assay:", count, "out of", len(assay_names))
#     df = pd.read_csv("assays/" + name, dtype = 'object')
#     output_file = open("output/" + name + ".txt", "a+")
#     for cid in cids:
#         cid_to_output[cid] = []
#         row = df.loc[df['PUBCHEM_CID'] == cid]
#         row = row.values.tolist()[0][9:] #first stuff is not important
#         #clean up data
#         for i in range(len(row)):
#             try:
#                 row[i] = float(row[i])
#             except:
#                 row[i] = 0 #should get filtered out by feature selection...
#             if math.isnan((row[i])): #emtpy cell, treat as 0 for neural network ease (bc inactive --> 0, right...?)
#                 row[i] = 0
#             row[i] = str(row[i])
#         cid_to_output[cid].append(row)
#         output_file.write(cid + ":" + " ".join(row) + "\n")


"""-------------------------------------------------------------------------------------------------------------------------------------"""
#What i did to go from output files to .npy files
# cids_file = open("all_important_cids.txt")
# whole_thing = cids_file.read()
# cids = whole_thing.split("\n")
# print("num important cids:", len(cids))
# cids_present = {}
# for cid in cids:
#     cids_present[float(cid)] = 1

# assay_names_file = open("important_assays.txt")
# whole_thing = assay_names_file.read()
# assay_names = whole_thing.split('\n')
# assay_names_file.close()

# output = []
# count = 0
# cid_to_output = {}
# for name in assay_names:
#     cids_logged = {}
#     count += 1
#     print("on assay", count, "out of", len(assay_names))
#     file = open("output/" + name + ".txt")
#     whole_thing = file.read()
#     file.close()
#     lines = whole_thing.split("\n")
#     for line in lines:
#         split = line.split(":")
#         if len(split) != 2:
#             continue
#         cid, data = float(split[0]), split[1].split()
#         data = np.array(data).astype(float)
#         if cid not in cids_present or cid in cids_logged: #not one of the cids we're studying or we've already got data on it
#             continue
#         cids_logged[cid] = 1 #got data on this cid
#         if cid not in cid_to_output:
#             cid_to_output[cid] = [data]
#         else:
#             cid_to_output[cid].append(data)

# cid_in_order = []
# print("number of cids:", len(cid_to_output))
# for cid in cid_to_output:
#     cid_in_order.append(cid)
#     output.append(cid_to_output[cid])

# #output dimensions: 6243 cids * 81 assays * 841-2,000ish datas (depending on assay)
# #official dimensions: 5076 x 81 x num_datas
# #len(output[0][0]) = 2040

# cid_in_order = np.asarray(cid_in_order, dtype=float)
# output = np.asarray(output, dtype=object)

# np.save("cid_in_order.npy", cid_in_order)
# print("saved cid in order")

# save output in separate files so they can be loaded within memory limits
# step = int(len(output)/10)
# a = 0
# b = step
# count = 0
# while (b < len(output)):
#     count += 1
#     temp = output[a:b]
#     np.save(open("output_part" + str(count) + ".npy", "wb+"), temp)
#     a += step
#     b += step
#     print("saved output part", count)

# last = output[b:]
# np.save(open("output_part" + str(count+1) + ".npy", "wb+"), last)
# print("success")