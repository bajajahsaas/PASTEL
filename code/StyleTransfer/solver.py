import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from utils import utilities as torch_utils
import numpy as np
import random
import math
import datetime
import gc
import sys
import pickle

from models.SeqToSeqAttn import SeqToSeqAttn


class Solver:

    def __init__(self, cnfg):
        torch.manual_seed(1)
        random.seed(7867567)
        styleProblems = ["STYLED", "ethnic", "gender", "education", "country", "tod", "politics"]
        styleProblems += [x + "-story" for x in styleProblems]
        if cnfg.problem in styleProblems:
            from utils import readData as readData
            cnfg.srcLang = "src"
            cnfg.tgtLang = "tgt"
            cnfg.taskInfo = cnfg.problem  # "STYLED"
        if cnfg.problem == "MT":
            from utils import readData as readData
            cnfg.srcLang = "de"
            cnfg.tgtLang = "en"
            cnfg.taskInfo = "en-de.low"

        cnfg.readData = readData
        self.cnfg = cnfg

        # Create Language Objects
        # self.cnfg.srcLangObj=self.cnfg.readData.Lang(self.cnfg.srcLang,self.cnfg.min_src_frequency)
        # self.cnfg.tgtLangObj=self.cnfg.readData.Lang(self.cnfg.tgtLang,self.cnfg.min_tgt_frequency)
        self.cnfg.srcLangObj = self.cnfg.readData.Lang(self.cnfg.srcLang, self.cnfg.min_src_frequency,
                                                       taskInfo=cnfg.taskInfo)
        self.cnfg.tgtLangObj = self.cnfg.readData.Lang(self.cnfg.tgtLang, self.cnfg.min_tgt_frequency,
                                                       taskInfo=cnfg.taskInfo)
        self.cnfg.srcLangObj.initVocab("train")
        self.cnfg.tgtLangObj.initVocab("train")
        if self.cnfg.share_embeddings:  # False by default
            self.cnfg.tgtLangObj.fuseVocab(self.cnfg.srcLangObj)

    def main(self):

        # Saving object variables as locals for quicker access
        cnfg = self.cnfg
        modelName = self.cnfg.modelName
        readData = self.cnfg.readData
        srcLangObj = self.cnfg.srcLangObj
        tgtLangObj = self.cnfg.tgtLangObj
        wids_src = srcLangObj.wids
        wids_tgt = tgtLangObj.wids

        # following has the list of lists. Each list is for one sentences (having word ids from dictionary)
        # using vocab from train while generating sentences for valid and test

        train_src = srcLangObj.read_corpus("train")
        train_tgt = tgtLangObj.read_corpus("train")
        if cnfg.mode != "inference" and cnfg.preTrain:
            preTrain_src = srcLangObj.read_corpus("pretrain")  # using train.STYLED.src vocab for pre-training mode also
            preTrain_tgt = tgtLangObj.read_corpus("pretrain")

        if cnfg.mode != "inference":
            valid_src = srcLangObj.read_corpus(mode="valid")
            valid_tgt = tgtLangObj.read_corpus(mode="valid")

        test_src = srcLangObj.read_corpus(mode="test")
        test_tgt = tgtLangObj.read_corpus(mode="test")

        train_src, train_tgt = train_src[:cnfg.max_train_sentences], train_tgt[:cnfg.max_train_sentences]
        print "src vocab size:", len(wids_src)
        print "tgt vocab size:", len(wids_tgt)
        print "training size:", len(train_src)
        '''
        src vocab size: 7783
        tgt vocab size: 8443
        training size: 33240
        valid size: 4155
        '''
        if cnfg.mode != "inference":
            print "valid size:", len(valid_src)

        train = zip(train_src, train_tgt)  # zip(train_src,train_tgt)
        if cnfg.mode != "inference":
            valid = zip(valid_src, valid_tgt)  # zip(train_src,train_tgt)

        train.sort(key=lambda x: len(x[0]))  # Sort based on length of source sentences (ascending). For batching

        if cnfg.mode != "inference":
            valid.sort(key=lambda x: len(x[0]))

        train_src, train_tgt = [x[0] for x in train], [x[1] for x in train]
        # x[0] is source component, x[1] is target component

        if cnfg.mode != "inference":
            valid_src, valid_tgt = [x[0] for x in valid], [x[1] for x in valid]

        # garbage default value is 3 and same is set to garbage token in readData.py L20
        train_src_batches, train_src_masks = torch_utils.splitBatches(train=train_src, batch_size=cnfg.batch_size,
                                                                      padSymbol=cnfg.garbage, method="pre")
        train_tgt_batches, train_tgt_masks = torch_utils.splitBatches(train=train_tgt, batch_size=cnfg.batch_size,
                                                                      padSymbol=cnfg.garbage, method="post")
        if cnfg.mode != "inference" and cnfg.preTrain:
            preTrain_src_batches, preTrain_src_masks = torch_utils.splitBatches(train=preTrain_src,
                                                                                batch_size=cnfg.batch_size,
                                                                                padSymbol=cnfg.garbage, method="pre")
            preTrain_tgt_batches, preTrain_tgt_masks = torch_utils.splitBatches(train=preTrain_tgt,
                                                                                batch_size=cnfg.batch_size,
                                                                                padSymbol=cnfg.garbage, method="post")

        if cnfg.mode != "inference":
            valid_src_batches, valid_src_masks = torch_utils.splitBatches(train=valid_src, batch_size=cnfg.batch_size,
                                                                          padSymbol=cnfg.garbage, method="pre")
            valid_tgt_batches, valid_tgt_masks = torch_utils.splitBatches(train=valid_tgt, batch_size=cnfg.batch_size,
                                                                          padSymbol=cnfg.garbage, method="post")

        test_src_batches, test_src_masks = torch_utils.splitBatches(train=test_src, batch_size=1,
                                                                    padSymbol=cnfg.garbage, method="pre")
        test_tgt_batches, test_tgt_masks = torch_utils.splitBatches(train=test_tgt, batch_size=1,
                                                                    padSymbol=cnfg.garbage, method="post")

        # Dump useless references
        train = None
        valid = None
        # Sanity check
        assert (len(train_tgt_batches) == len(train_src_batches))
        if cnfg.mode != "inference":
            assert (len(valid_tgt_batches) == len(valid_src_batches))
        print(len(test_tgt_batches), len(test_src_batches))  # (4155, 4155)
        assert (len(test_tgt_batches) == len(test_src_batches))
        '''
        Training Batches: 1039
        Validation Batches: 130
        '''
        print "Training Batches:", len(train_tgt_batches)
        if cnfg.mode != "inference":
            print "Validation Batches:", len(valid_tgt_batches)
        print "Test Points:", len(test_src_batches)  # Test Points: 4155

        if cnfg.cudnnBenchmark:
            torch.backends.cudnn.benchmark = True
        # Declare model object
        print "Declaring Model, Loss, Optimizer"
        model = SeqToSeqAttn(cnfg, wids_src=wids_src, wids_tgt=wids_tgt)
        loss_function = nn.NLLLoss(ignore_index=1, size_average=False)
        if torch.cuda.is_available():
            model.cuda()
            loss_function = loss_function.cuda()
        optimizer = None
        if cnfg.optimizer_type == "SGD":
            optimizer = optim.SGD(model.getParams(), lr=0.05)
        elif cnfg.optimizer_type == "ADAM":  # default
            optimizer = optim.Adam(model.getParams())

        if cnfg.mode == "trial":
            print "Running Sample Batch"
            print "Source Batch Shape:", train_src_batches[30].shape
            print "Source Mask Shape:", train_src_masks[30].shape
            print "Target Batch Shape:", train_tgt_batches[30].shape
            print "Target Mask Shape:", train_tgt_masks[30].shape
            sample_src_batch = train_src_batches[30]
            sample_tgt_batch = train_tgt_batches[30]
            sample_mask = train_tgt_masks[30]
            sample_src_mask = train_src_masks[30]
            print datetime.datetime.now()
            model.zero_grad()
            loss = model.forward(sample_src_batch, sample_tgt_batch, sample_src_mask, sample_mask, loss_function)
            print loss
            loss.backward()
            optimizer.step()
            print datetime.datetime.now()
            print "Done Running Sample Batch"

        train_batches = zip(train_src_batches, train_tgt_batches, train_src_masks, train_tgt_masks)
        if cnfg.mode != "inference":
            valid_batches = zip(valid_src_batches, valid_tgt_batches, valid_src_masks, valid_tgt_masks)
        if cnfg.mode != "inference" and cnfg.preTrain:
            preTrain_batches = zip(preTrain_src_batches, preTrain_tgt_batches, preTrain_src_masks, preTrain_tgt_masks)

        train_src_batches, train_tgt_batches, train_src_masks, train_tgt_masks = None, None, None, None
        if cnfg.mode != "inference":
            valid_src_batches, valid_tgt_batches, valid_src_masks, valid_tgt_masks = None, None, None, None

        if cnfg.mode == "train" or cnfg.mode == "LM":
            print "Start Time:", datetime.datetime.now()
            if cnfg.preTrain:
                preTrainOptimizer = None
                if cnfg.optimizer_type == "SGD":
                    preTrainOptimizer = optim.SGD(model.getParams(), lr=0.05)
                elif cnfg.optimizer_type == "ADAM":
                    preTrainOptimizer = optim.Adam(model.getParams())

                for epochId in range(cnfg.NUM_PRETRAIN_EPOCHS):
                    random.shuffle(preTrain_batches)
                    for batchId, batch in enumerate(preTrain_batches):
                        src_batch, tgt_batch, src_mask, tgt_mask = batch[0], batch[1], batch[2], batch[3]
                        batchLength = src_batch.shape[1]
                        batchSize = src_batch.shape[0]
                        tgtBatchLength = tgt_batch.shape[1]
                        if batchLength < cnfg.MAX_SEQ_LEN and batchSize > 1 and tgtBatchLength < cnfg.MAX_TGT_SEQ_LEN:
                            model.zero_grad()
                            loss = model.forward(src_batch, tgt_batch, src_mask, tgt_mask, loss_function)
                            if cnfg.mem_optimize:
                                del src_batch, tgt_batch, src_mask, tgt_mask
                            loss.backward()
                            if cnfg.mem_optimize:
                                del loss
                            preTrainOptimizer.step()
                        if batchId % cnfg.PRINT_STEP == 0:
                            print "Batch No:", batchId, " Time:", datetime.datetime.now()
                    print "Pre-Training Epoch:", epochId

            for epochId in range(cnfg.NUM_EPOCHS):
                random.shuffle(train_batches)
                for batchId, batch in enumerate(train_batches):
                    src_batch, tgt_batch, src_mask, tgt_mask = batch[0], batch[1], batch[2], batch[3]
                    batchLength = src_batch.shape[1]
                    batchSize = src_batch.shape[0]
                    tgtBatchLength = tgt_batch.shape[1]
                    if batchLength < cnfg.MAX_SEQ_LEN and batchSize > 1 and tgtBatchLength < cnfg.MAX_TGT_SEQ_LEN:
                        model.zero_grad()
                        loss = model.forward(src_batch, tgt_batch, src_mask, tgt_mask, loss_function)
                        if cnfg.mem_optimize:
                            del src_batch, tgt_batch, src_mask, tgt_mask
                        loss.backward()
                        if cnfg.mem_optimize:
                            del loss
                        optimizer.step()
                    if batchId % cnfg.PRINT_STEP == 0:
                        print "Batch No:", batchId, " Time:", datetime.datetime.now()

                totalValidationLoss = 0.0
                NUM_TOKENS = 0.0
                for batchId, batch in enumerate(valid_batches):
                    src_batch, tgt_batch, src_mask, tgt_mask = batch[0], batch[1], batch[2], batch[3]
                    batchSize = src_batch.shape[0]
                    if batchSize <= 1:
                        continue
                    model.zero_grad()
                    loss = model.forward(src_batch, tgt_batch, src_mask, tgt_mask, loss_function, inference=True)
                    if cnfg.normalizeLoss:
                        totalValidationLoss += (loss.data.cpu().numpy()) * np.sum(tgt_mask)
                    else:
                        totalValidationLoss += (loss.data.cpu().numpy())
                    NUM_TOKENS += np.sum(tgt_mask)
                    if cnfg.mem_optimize:
                        del src_batch, tgt_batch, src_mask, tgt_mask, loss

                model.save_checkpoint(modelName + "_" + str(epochId), optimizer)

                perplexity = math.exp(totalValidationLoss / NUM_TOKENS)
                print "Epoch:", epochId, " Total Validation Loss:", totalValidationLoss, " Perplexity:", perplexity
            print "End Time:", datetime.datetime.now()

        elif cnfg.mode == "inference":
            if cnfg.method == "OSOM":
                import levenshtein as levenshtein
                train_src = train_src[:10000]  # [:500]
                train_tgt = train_tgt[:10000]  # [:500]
                trainIndex = {}
                for i in range(len(train_src)):
                    trainIndex[i] = (train_src[i], train_tgt[i])
                testIndex = {}
                fineTuneBatches = {}
                for i in range(len(test_src)):
                    if i % 300 == 0:
                        print "Computed Similarity Upto:", i
                    simValues = []
                    for j in trainIndex:
                        simValue = levenshtein.levenshtein(trainIndex[j][0], test_src[i])
                        simValues.append((j, simValue))
                    simValues.sort(key=lambda x: x[1])
                    simValues = simValues[:4]
                    # print simValues
                    simValues = [x[0] for x in simValues]
                    if len(simValues) % 2 == 1:
                        # If odd, make it even by giving double importance to most similar sentence.
                        simValues.append(simValues[0])
                    train_src = [trainIndex[x][0] for x in simValues]
                    train_tgt = [trainIndex[x][1] for x in simValues]

                    # print i,":",simValues
                    train_src_batches, train_src_masks = torch_utils.splitBatches(train=train_src,
                                                                                  batch_size=len(train_src),
                                                                                  padSymbol=cnfg.garbage, method="pre")
                    train_tgt_batches, train_tgt_masks = torch_utils.splitBatches(train=train_tgt,
                                                                                  batch_size=len(train_src),
                                                                                  padSymbol=cnfg.garbage, method="post")
                    testIndex[i] = zip(train_src_batches, train_tgt_batches, train_src_masks, train_tgt_masks)
                    # print testIndex[i][0]
                    # print testIndex[i][1]
                print "Done loading similarity matrix"
                model.load_from_checkpoint(modelName)
                model.decodeAll(test_src_batches, modelName, method=cnfg.method, evalMethod="BLEU", suffix="test",
                                testIndex=testIndex, loss_function=loss_function, optimizer=optimizer)
                exit()
            model.load_from_checkpoint(modelName)
            "Loaded Model"
            "Saving Embeddings and Vocabulary"
            encodeEmbed = model.encoder.embeddings.weight.data.cpu().numpy()
            decodeEmbed = model.decoder.embeddings.weight.data.cpu().numpy()
            encodeVocab = {}
            encodeReverseVocab = {}
            for key, val in model.wids_src.items():
                encodeVocab[key] = val
                encodeReverseVocab[val] = key
            decodeVocab = {}
            decodeReverseVocab = {}
            for key, val in model.wids_tgt.items():
                decodeVocab[key] = val
                decodeReverseVocab[val] = key

            pickle.dump(encodeEmbed, open(modelName + ".encodeEmbed", "wb"))
            pickle.dump(encodeVocab, open(modelName + ".encodeVocab", "wb"))
            pickle.dump(encodeReverseVocab, open(modelName + ".encodeReverseVocab", "wb"))
            pickle.dump(decodeEmbed, open(modelName + ".decodeEmbed", "wb"))
            pickle.dump(decodeVocab, open(modelName + ".decodeVocab", "wb"))
            pickle.dump(decodeReverseVocab, open(modelName + ".decodeReverseVocab", "wb"))
            "Finished Saving"
            # Evaluate on test first
            model.decodeAll(test_src_batches, modelName, method=cnfg.method, evalMethod="BLEU", suffix="test",
                            lmObj=self.cnfg.lmObj, getAtt=self.cnfg.getAtt)
            # Also on valid

            # valid_src=srcLangObj.read_corpus(mode="valid")
            # valid_src_batches,valid_src_masks=torch_utils.splitBatches(train=valid_src,batch_size=1,padSymbol=cnfg.garbage,method="pre")
            # model.decodeAll(valid_src_batches,modelName,method=cnfg.method,evalMethod="BLEU",suffix="valid",lmObj=self.cnfg.lmObj)
