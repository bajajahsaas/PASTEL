import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import numpy as np
import random
import math
import datetime
import gc
import sys
import pickle
from modules import *
from embeddingUtils import *
import itertools


class SeqToSeqAttn():
    def __init__(self, cnfg, wids_src=None, wids_tgt=None):

        self.wids_src = wids_src
        self.wids_tgt = wids_tgt
        self.reverse_wids_src = torch_utils.reverseDict(wids_src)
        self.reverse_wids_tgt = torch_utils.reverseDict(wids_tgt)
        self.cnfg = cnfg
        self.cnfg.srcVocabSize = len(self.wids_src)
        self.cnfg.tgtVocabSize = len(self.wids_tgt)

        #  emb_size=300, hidden_size=384, use_LSTM = True, share_embeddings = False
        self.encoder = EncoderRNN(self.wids_src, self.cnfg.srcVocabSize, self.cnfg.emb_size, self.cnfg.hidden_size,
                                  self.cnfg.use_LSTM, False)
        if self.cnfg.use_reverse:  # use_reverse = True
            #  share embeddings between forward and reverse encoders
            self.revcoder = EncoderRNN(self.wids_src, self.cnfg.srcVocabSize, self.cnfg.emb_size, self.cnfg.hidden_size,
                                       self.cnfg.use_LSTM, True, reference_embeddings=self.encoder.embeddings)
        # share_embeddings is False, no use of reference_embeddings here. sigmoid is False too
        self.decoder = AttnDecoderRNN(self.wids_tgt, cnfg.tgtVocabSize, cnfg.emb_size, cnfg.hidden_size, cnfg.use_LSTM,
                                      cnfg.use_attention, cnfg.share_embeddings, self.cnfg.pointer,
                                      reference_embeddings=self.encoder.embeddings, sigmoid=self.cnfg.sigmoid)

        if self.cnfg.initGlove or self.cnfg.initGlove2:
            embed = self.decoder.embeddings.weight.data.cpu().numpy()
            if self.cnfg.initGlove:
                embed = loadEmbedsAsNumpyObj("./glove/wiki.simple.vec", self.wids_tgt, embed)
            elif self.cnfg.initGlove2:
                embed = loadEmbedsAsNumpyObj("../nlg-eval/nlgeval/data/glove.6B.300d.txt", self.wids_tgt, embed)
            self.decoder.embeddings.weight.data.copy_(torch.from_numpy(embed))

        if self.cnfg.initGloveEncode or self.cnfg.initGloveEncode2:
            embed = self.encoder.embeddings.weight.data.cpu().numpy()
            if self.cnfg.initGloveEncode:
                embed = loadEmbedsAsNumpyObj("./glove/wiki.simple.vec", self.wids_src, embed)
            elif self.cnfg.initGloveEncode2:
                embed = loadEmbedsAsNumpyObj("../nlg-eval/nlgeval/data/glove.6B.300d.txt", self.wids_src, embed)

            self.encoder.embeddings.weight.data.copy_(torch.from_numpy(embed))

        if self.cnfg.use_attention and self.cnfg.use_downstream:  # Both True
            self.W = LinearLayer(2 * self.cnfg.hidden_size, self.cnfg.tgtVocabSize)
            # concatenate attention output (weighted sum of encoder hidden states) and hidden state of decoder
        else:
            self.W = LinearLayer(self.cnfg.hidden_size, self.cnfg.tgtVocabSize)

        if self.cnfg.pointer:  # pointer only possible when encoder side attention is true
            self.ptr = LinearLayer(2 * self.cnfg.hidden_size, 1)

        if self.cnfg.embeddingFreeze:  # False
            # self.encoder.embeddings.weight.requires_grad=False
            # self.revcoder.embeddings.weight.requires_grad=False
            self.decoder.embeddings.weight.requires_grad = False

    def zero_grad(self):
        self.encoder.zero_grad()
        self.revcoder.zero_grad()
        self.decoder.zero_grad()
        self.W.zero_grad()

    def cuda(self):
        self.encoder.cuda()
        self.revcoder.cuda()
        self.decoder.cuda()
        self.W.cuda()
        if self.cnfg.pointer:
            self.ptr.cuda()

    def getIndex(self, row, inference=False):
        tensor = torch.LongTensor(row)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return autograd.Variable(tensor, volatile=inference)

    def init_hidden(self, batch):
        hiddenElem1 = torch.zeros(1, batch.shape[1], self.cnfg.hidden_size)
        if self.cnfg.use_LSTM:
            hiddenElem2 = torch.zeros(1, batch.shape[1], self.cnfg.hidden_size)
        if torch.cuda.is_available():
            hiddenElem1 = hiddenElem1.cuda()
            if self.cnfg.use_LSTM:
                hiddenElem2 = hiddenElem2.cuda()
        if self.cnfg.use_LSTM:
            return (autograd.Variable(hiddenElem1), autograd.Variable(hiddenElem2))
        else:
            return autograd.Variable(hiddenElem1)

    def save_checkpoint(self, modelName, optimizer):
        if not self.cnfg.pointer:
            checkpoint = {'encoder_state_dict': self.encoder.state_dict(),
                          'revcoder_state_dict': self.revcoder.state_dict(),
                          'decoder_state_dict': self.decoder.state_dict(), 'lin_dict': self.W.state_dict(),
                          'optimizer': optimizer.state_dict()}
        else:
            checkpoint = {'encoder_state_dict': self.encoder.state_dict(),
                          'revcoder_state_dict': self.revcoder.state_dict(),
                          'decoder_state_dict': self.decoder.state_dict(), 'lin_dict': self.W.state_dict(),
                          'ptr_dict': self.ptr.state_dict(),
                          'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, self.cnfg.model_dir + modelName + ".ckpt")
        print "Saved Model"
        return

    def getParams(self):
        all_params = list(self.encoder.parameters()) + list(self.revcoder.parameters()) + list(
            self.decoder.parameters()) + list(self.W.parameters())

        if self.cnfg.embeddingFreeze:
            all_params = itertools.ifilter(lambda p: p.requires_grad, all_params)

        return all_params

    def load_from_checkpoint(self, modelName, optimizer=None):
        checkpoint = torch.load(modelName)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.revcoder.load_state_dict(checkpoint['revcoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.W.load_state_dict(checkpoint['lin_dict'])

        if self.cnfg.pointer:
            self.ptr.load_state_dict(checkpoint['ptr_dict'])

        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print "Loaded Model"
        return

    def decodeAll(self, srcBatches, modelName, method="greedy", evalMethod="BLEU", suffix="test", lmObj=None,
                  testIndex=None, loss_function=None, optimizer=None, getAtt=False):
        # test_src_batches, modelName, method="BEAM", evalMethod="BLEU", suffix="test",
        #                             lmObj=None, getAtt=False
        tgtStrings = []
        tgtTimes = []
        if getAtt:
            tgtAtts = []
        totalTime = 0.0
        print "Decoding Start Time:", datetime.datetime.now()
        for i, srcBatch in enumerate(srcBatches):
            # iterate each source sentence in test_set and make it stylised
            tgtString = None
            startTime = datetime.datetime.now()
            # print(method)
            if method == "greedy":
                if not getAtt:
                    tgtString = self.greedyDecode(srcBatch)
                else:
                    tgtString, tgtAtt = self.greedyDecode(srcBatch, getAtt=True)
            elif method == "beam":
                tgtString = self.beamDecode(srcBatch)
            elif method == "topk":
                tgtString = self.samplingDecode(srcBatch)
            elif method == "beamLM":
                tgtString = self.beamDecode(srcBatch, useLM=True, lmObj=lmObj)
            elif method == "beamSib":
                tgtString = self.beamDecode(srcBatch, beamSib=True)
            elif method == "OSOM":
                print "Loading Model Again"
                self.load_from_checkpoint(modelName, optimizer=optimizer)
                fine_batches = testIndex[i]
                print "Fine-Tuning Model for", i
                for batchId, batch in enumerate(fine_batches):
                    src_batch, tgt_batch, src_mask, tgt_mask = batch[0], batch[1], batch[2], batch[3]
                    self.zero_grad()
                    loss = self.forward(src_batch, tgt_batch, src_mask, tgt_mask, loss_function, inference=False)
                    del src_batch, tgt_batch, src_mask, tgt_mask
                    loss.backward()
                    del loss
                    optimizer.step()
                    # print "Update for Fine-Tuning BatchId:",batchId
                print "Finished Fine-Tuning Model"
                tgtString = self.greedyDecode(srcBatch)

            endTime = datetime.datetime.now()
            timeTaken = (endTime - startTime).total_seconds()
            totalTime += timeTaken
            if i % 100 == 0:
                print "Decoding Example ", i, " Time Taken ", timeTaken
            tgtTimes.append(timeTaken)
            tgtStrings.append(tgtString)
            if getAtt:
                tgtAtts.append(tgtAtt)
        print "Decoding End Time:", datetime.datetime.now()
        print "Total Decoding Time:", totalTime

        # Dump Output
        if method == "greedy":
            outFileName = modelName + "." + suffix + ".output"
            if getAtt:
                attFileName = modelName + "." + suffix + ".att"
        else:
            outFileName = modelName + "." + suffix + "." + method + ".output"

        outFile = open(outFileName, "w")
        for tgtString in tgtStrings:
            outFile.write(tgtString + "\n")
        outFile.close()

        if getAtt:
            pickle.dump(tgtAtts, open(attFileName, "wb"))

        # Dump Times
        timeFileName = modelName + "." + suffix + ".time"
        timeFile = open(timeFileName, "w")
        for tgtTime in tgtTimes:
            timeFile.write(str(tgtTime) + "\n")
        timeFile.close()

        if evalMethod == "BLEU":  # True
            import os
            if self.cnfg.problem == "MT":  # False
                BLEUOutput = os.popen(
                    "perl multi-bleu.perl -lc " + "data/" + suffix + ".en-de.low.en" + " < " + outFileName).read()
            elif self.cnfg.problem == "CHESS":  # False
                BLEUOutput = os.popen(
                    "perl multi-bleu.perl -lc " + "data/" + suffix + ".che-eng.single.en" + " < " + outFileName).read()

            print BLEUOutput
        # Compute BLEU
        elif evalMethod == "ROUGE":  # False
            print "To implement ROUGE"

        return tgtStrings

    def beamDecode(self, srcBatch, useLM=False, lmObj=None, beamSib=False):
        #  test set (source sentences), useLM = False, lmObj = None
        k = self.cnfg.beamSize  # default = 3

        # srcBatch : batch_size x seqlen
        batch_size = srcBatch.shape[0]
        srcSentenceLength = srcBatch.shape[1]
        srcBatch = srcBatch.T  # seqlen x batchsize
        self.enc_hidden = self.init_hidden(srcBatch)
        enc_out = None
        encoderOuts = []
        if self.cnfg.use_reverse:
            self.rev_hidden = self.init_hidden(srcBatch)
            rev_out = None
            revcoderOuts = []
        srcEmbedIndexSeq = []
        for rowId, row in enumerate(srcBatch):
            srcEmbedIndex = self.getIndex(row, inference=True)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out, self.enc_hidden = self.encoder(srcBatch.shape[1], srcEmbedIndex, self.enc_hidden)
            encoderOuts.append(enc_out.view(1, -1))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out, self.rev_hidden = self.revcoder(srcBatch.shape[1], srcEmbedIndex, self.rev_hidden)

                revcoderOuts.append(rev_out.view(1, -1))
            revcoderOuts.reverse()

        if self.cnfg.use_reverse:
            encoderOuts = [torch.add(x, y) for x, y in zip(encoderOuts, revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del srcBatch
            del enc_out

        zeroInit = torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit = zeroInit.cuda()
        c_0 = autograd.Variable(zeroInit)

        self.hidden = self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed == False:
                if self.cnfg.init_enc:
                    self.hidden = self.enc_hidden
                else:
                    self.hidden = self.rev_hidden
            else:
                if self.cnfg.use_LSTM:
                    self.hidden = (torch.add(self.enc_hidden[0], self.rev_hidden[0]),
                                   torch.add(self.enc_hidden[1], self.rev_hidden[1]))
                else:
                    self.hidden = torch.add(self.enc_hidden, self.rev_hidden)

        tgts = []
        row = np.array([self.cnfg.start, ] * 1)

        tgtEmbedIndex = self.getIndex(row, inference=True)

        out, self.hidden, c_0 = self.decoder(1, tgtEmbedIndex, None, None, self.hidden, feedContextVector=True,
                                             contextVector=c_0)
        # forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)

        out = out.view(1, -1)
        if self.cnfg.use_attention:
            scores = self.W(torch.cat([out, c_0], 1))
        else:
            scores = self.W(out)

        maxValues, argmaxes = torch.max(scores, 1)
        argmaxValue = argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            if not (self.cnfg.decoder_prev_random or self.cnfg.mixed_decoding):
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        encOutTensor = torch.cat([encoderOut.view(1, 1, self.cnfg.hidden_size) for encoderOut in encoderOuts], 1)

        beams = [(self.hidden, out, 0.0, [tgts[0], ],
                  False)]  # Current state, current output, current score, current tgts, stopped boolean
        completedBeams = []  # Stop when this reaches k.

        steps = 0
        while len(completedBeams) < k and steps < 2 * srcSentenceLength + 10:  # self.cnfg.TGT_LEN_LIMIT:
            # print "Step ",steps
            expandedBeams = []
            for beam in beams:
                if beam[4]:
                    continue
                row = np.array([beam[3][-1], ] * 1)
                tgtEmbedIndex = self.getIndex(row, inference=True)
                o_t = beam[1]  # out
                # print np.shape(row)
                # print tgtEmbedIndex.size()
                # print o_t.size()
                # print beam[0][0].size()
                # print beam[0][1].size()
                # print encOutTensor.size()
                if not self.cnfg.pointer:
                    out, newHidden, c_t = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t,
                                                       beam[0],
                                                       feedContextVector=False, inference=True)
                else:
                    out, newHidden, c_t, a_t = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t,
                                                            beam[0],
                                                            feedContextVector=False, inference=True)
                del o_t

                out = out.view(1, -1)
                if self.cnfg.use_attention:
                    if not self.cnfg.pointer:
                        scores = F.log_softmax(self.W(torch.cat([out, c_t], 1)))
                    else:
                        srcBatch_tensor = torch.from_numpy(srcBatch)
                        if torch.cuda.is_available():
                            srcBatch_tensor = srcBatch_tensor.cuda()
                        # dim: seqlen x batch_size

                        logits = torch.cat([out, c_t], 1)
                        output = torch.zeros(batch_size, self.cnfg.tgtVocabSize)
                        if torch.cuda.is_available():
                            output = output.cuda()

                        # distribute probabilities between generator and pointer
                        prob_ptr_logits = self.ptr(logits)
                        prob_ptr = F.sigmoid(prob_ptr_logits)  # (batch size, 1)
                        prob_gen = 1 - prob_ptr
                        # add generator probabilities to output
                        gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
                        output[:, :self.cnfg.tgtVocabSize] = prob_gen * gen_output
                        # using source side vocab to generate words
                        # add pointer probabilities to output
                        ptr_output = a_t
                        # batchsize x seq_len
                        output.scatter_add_(1, srcBatch_tensor.transpose(0, 1), prob_ptr * ptr_output)
                        scores = torch.log(output + 1e-31)

                else:
                    scores = F.log_softmax(self.W(out))

                maxValues, argmaxes = torch.topk(scores, k=k, dim=1)
                argmaxValues = argmaxes.cpu().data.numpy()
                maxValues = maxValues.cpu().data.numpy()
                for kprime in range(k):
                    argmaxValue = argmaxValues[:, kprime][0]
                    maxValue = maxValues[:, kprime][0]
                    newScore = beam[2] + maxValue  # -0.5*kprime
                    if beamSib:
                        newScore = beam[2] + maxValue - 0.5 * kprime
                    entry = (newHidden, out, newScore, list(beam[3]) + [argmaxValue, ], False)
                    if argmaxValue == self.cnfg.stop:
                        modifiedEntry = (newHidden, out, newScore, list(beam[3]) + [argmaxValue, ], True)
                        completedBeams.append(entry)
                        # print "Completed Beam: ",len(completedBeams)
                    else:
                        expandedBeams.append(entry)
                        # print "Expanded Beam: ",len(expandedBeams)
                # argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
                # tgts.append(argmaxValue)

                # Add to expandedBeams
                # Add stopped beams to completedBeams

            expandedBeams.sort(key=lambda x: -x[2])  # Sort the expanded beams
            beams = expandedBeams[:k]  # Keep top k for next iteration
            steps += 1

        # Put remaining beams into expanded Beams

        # Filter out overtly short beams
        newExpandedBeams = []
        for beam in completedBeams + beams:
            if len(beam[3]) >= 3:
                # beam[2]=beam[2]-0.05*lmObj.score(beam[3])
                newExpandedBeams.append(beam)

        # print "Final Number of Expanded Beams:",len(newExpandedBeams)
        if useLM:
            newExpandedBeams.sort(key=lambda x: -x[2] / len(x[3]) - 0.05 * lmObj.score(x[3][:min(5, len(x[3]))]))
        else:
            newExpandedBeams.sort(key=lambda x: -x[2] / len(x[3]))

        tgts = newExpandedBeams[0][3]
        if tgts[-1] == self.cnfg.stop:
            tgts = tgts[:-1]

        return " ".join([self.reverse_wids_tgt[x] for x in tgts])

    def samplingDecode(self, srcBatch):
        
        k = self.cnfg.beamSize  # default = 3

        # srcBatch : batch_size x seqlen
        batch_size = srcBatch.shape[0]
        srcSentenceLength = srcBatch.shape[1]
        srcBatch = srcBatch.T  # seqlen x batchsize

        self.enc_hidden = self.init_hidden(srcBatch)
        enc_out = None
        encoderOuts = []

        if self.cnfg.use_reverse:
            self.rev_hidden = self.init_hidden(srcBatch)
            rev_out = None
            revcoderOuts = []

        srcEmbedIndexSeq = []
        for rowId, row in enumerate(srcBatch):
            srcEmbedIndex = self.getIndex(row, inference=True)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out, self.enc_hidden = self.encoder(srcBatch.shape[1], srcEmbedIndex, self.enc_hidden)
            encoderOuts.append(enc_out.view(1, -1))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out, self.rev_hidden = self.revcoder(srcBatch.shape[1], srcEmbedIndex, self.rev_hidden)

                revcoderOuts.append(rev_out.view(1, -1))
            revcoderOuts.reverse()
        
        if self.cnfg.use_reverse:
            encoderOuts = [torch.add(x, y) for x, y in zip(encoderOuts, revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del srcBatch
            del enc_out

        zeroInit = torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit = zeroInit.cuda()
        c_0 = autograd.Variable(zeroInit)

        self.hidden = self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed == False:
                if self.cnfg.init_enc:
                    self.hidden = self.enc_hidden
                else:
                    self.hidden = self.rev_hidden
            else:
                if self.cnfg.use_LSTM:
                    self.hidden = (torch.add(self.enc_hidden[0], self.rev_hidden[0]),torch.add(self.enc_hidden[1], self.rev_hidden[1]))
                else:
                    self.hidden = torch.add(self.enc_hidden, self.rev_hidden)

        tgts = []
        
        row = np.array([self.cnfg.start, ] * 1)

        tgtEmbedIndex = self.getIndex(row, inference=True)

        out, self.hidden, c_0 = self.decoder(1, tgtEmbedIndex, None, None, self.hidden, feedContextVector=True, contextVector=c_0)
        # forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)

        out = out.view(1, -1)
        if self.cnfg.use_attention:
            scores = self.W(torch.cat([out, c_0], 1))
        else:
            scores = self.W(out)

        maxValues, argmaxes = torch.max(scores, 1)
        argmaxValue = argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            if not (self.cnfg.decoder_prev_random or self.cnfg.mixed_decoding):
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        encOutTensor = torch.cat([encoderOut.view(1, 1, self.cnfg.hidden_size) for encoderOut in encoderOuts], 1)

        while argmaxValue != self.cnfg.stop and len(tgts) < 2 * srcSentenceLength + 10:  # self.cnfg.TGT_LEN_LIMIT:
            print "iteration #", len(tgts)
            row = np.array([argmaxValue, ] * 1)
            tgtEmbedIndex = self.getIndex(row, inference=True)
            o_t = out  # out
            # print np.shape(row)
            # print tgtEmbedIndex.size()
            # print o_t.size()
            # print encOutTensor.size()
            if not self.cnfg.pointer:
                out, newHidden, c_t = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t, self.hidden, feedContextVector=False, inference=True)
            else:
                out, newHidden, c_t, a_t = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t, self.hidden, feedContextVector=False, inference=True)
            del o_t

            out = out.view(1, -1)
            if self.cnfg.use_attention:
                if not self.cnfg.pointer:
                    scores = F.softmax(self.W(torch.cat([out, c_t], 1)))
                else:
                    srcBatch_tensor = torch.from_numpy(srcBatch)
                    if torch.cuda.is_available():
                        srcBatch_tensor = srcBatch_tensor.cuda()
                        # dim: seqlen x batch_size

                    logits = torch.cat([out, c_t], 1)
                    output = torch.zeros(batch_size, self.cnfg.tgtVocabSize)
                    if torch.cuda.is_available():
                        output = output.cuda()

                    # distribute probabilities between generator and pointer
                    prob_ptr_logits = self.ptr(logits)
                    prob_ptr = F.sigmoid(prob_ptr_logits)  # (batch size, 1)
                    prob_gen = 1 - prob_ptr
                    # add generator probabilities to output
                    gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
                    output[:, :self.cnfg.tgtVocabSize] = prob_gen * gen_output
                    # using source side vocab to generate words
                    # add pointer probabilities to output
                    ptr_output = a_t
                    # batchsize x seq_len
                    output.scatter_add_(1, srcBatch_tensor.transpose(0, 1), prob_ptr * ptr_output)
                    scores = output

            else:
                scores = F.softmax(self.W(out))

            maxValues, argmaxes = torch.topk(scores, k=k, dim=1)
            print "top k shape", argmaxes.shape
            argmaxValues = argmaxes.cpu().squeeze().data.numpy()
            maxValues = maxValues.cpu().squeeze().data.numpy()
            
            maxValues /= maxValues.sum()

            argmaxValue = np.random.choice(argmaxValues, 1, p = maxValues)
            print argmaxValue
            tgts.append(argmaxValue)
        
        if tgts[-1] == self.cnfg.stop:
            tgts = tgts[:-1]

        return " ".join([self.reverse_wids_tgt[x] for x in tgts])

    def greedyDecode(self, srcBatch, getAtt=False):
        # Note: srcBatch is of size 1
        srcSentenceLength = srcBatch.shape[1]
        srcBatch = srcBatch.T
        self.enc_hidden = self.init_hidden(srcBatch)
        enc_out = None
        encoderOuts = []
        if self.cnfg.use_reverse:
            self.rev_hidden = self.init_hidden(srcBatch)
            rev_out = None
            revcoderOuts = []

        srcEmbedIndexSeq = []
        for rowId, row in enumerate(srcBatch):
            srcEmbedIndex = self.getIndex(row, inference=True)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out, self.enc_hidden = self.encoder(srcBatch.shape[1], srcEmbedIndex, self.enc_hidden)
            encoderOuts.append(enc_out.view(1, -1))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out, self.rev_hidden = self.revcoder(srcBatch.shape[1], srcEmbedIndex, self.rev_hidden)

                revcoderOuts.append(rev_out.view(1, -1))
            revcoderOuts.reverse()

        if self.cnfg.use_reverse:
            encoderOuts = [torch.add(x, y) for x, y in zip(encoderOuts, revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            if not getAtt:
                del srcBatch
            del enc_out

        zeroInit = torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit = zeroInit.cuda()
        c_0 = autograd.Variable(zeroInit)

        self.hidden = self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed == False:
                if self.cnfg.init_enc:
                    self.hidden = self.enc_hidden
                else:
                    self.hidden = self.rev_hidden
            else:
                if self.cnfg.use_LSTM:
                    self.hidden = (torch.add(self.enc_hidden[0], self.rev_hidden[0]),
                                   torch.add(self.enc_hidden[1], self.rev_hidden[1]))
                else:
                    self.hidden = torch.add(self.enc_hidden, self.rev_hidden)

        tgts = []
        if getAtt:
            atts = []
        row = np.array([self.cnfg.start, ] * 1)

        tgtEmbedIndex = self.getIndex(row, inference=True)

        out, self.hidden, c_0 = self.decoder(1, tgtEmbedIndex, None, None, self.hidden, feedContextVector=True,
                                             contextVector=c_0)
        # forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)

        out = out.view(1, -1)
        if self.cnfg.use_attention:
            scores = self.W(torch.cat([out, c_0], 1))
        else:
            scores = self.W(out)

        maxValues, argmaxes = torch.max(scores, 1)
        argmaxValue = argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            if not (self.cnfg.decoder_prev_random or self.cnfg.mixed_decoding):
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        encOutTensor = torch.cat([encoderOut.view(1, 1, self.cnfg.hidden_size) for encoderOut in encoderOuts], 1)
        while argmaxValue != self.cnfg.stop and len(tgts) < 2 * srcSentenceLength + 10:  # self.cnfg.TGT_LEN_LIMIT:
            row = np.array([argmaxValue, ] * 1)
            tgtEmbedIndex = self.getIndex(row, inference=True)
            o_t = out

            if not getAtt:
                out, self.hidden, c_t = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t,
                                                     self.hidden, feedContextVector=False, inference=True)
            else:
                out, self.hidden, c_t, att = self.decoder(1, tgtEmbedIndex, torch.transpose(encOutTensor, 0, 1), o_t,
                                                          self.hidden, feedContextVector=False, inference=True,
                                                          getAtt=True)

            del o_t

            out = out.view(1, -1)
            if self.cnfg.use_attention:
                scores = self.W(torch.cat([out, c_t], 1))
            else:
                scores = self.W(out)

            maxValues, argmaxes = torch.max(scores, 1)
            argmaxValue = argmaxes.view(1).cpu().data.numpy()[0]
            tgts.append(argmaxValue)
            if getAtt:
                atts.append(att)

        if tgts[-1] == self.cnfg.stop:
            tgts = tgts[:-1]
            if getAtt:
                atts = atts[:-1]

        if not getAtt:
            return " ".join([self.reverse_wids_tgt[x] for x in tgts])
        else:
            return " ".join([self.reverse_wids_tgt[x] for x in tgts]), [atts, srcBatch, tgts]

    def forward(self, srcBatch, batch, srcMask, mask, loss_function, inference=False):
        #  srcBatch: batchsize x timestamps (batchsize is the number of sentences in that batch)
        srcBatch = srcBatch.T
        srcMask = srcMask.T
        # timestamps x batchsize

        # Init encoder. We don't need start here since we don't softmax.
        self.enc_hidden = self.init_hidden(srcBatch)  # 1 x batchsize x hidden_dim
        # print "Src Batch Size:",srcBatch.shape
        # print "Src Mask Size:",srcMask.shape

        enc_out = None
        encoderOuts = []

        if self.cnfg.use_reverse:
            self.rev_hidden = self.init_hidden(srcBatch)
            rev_out = None
            revcoderOuts = []

        srcEmbedIndexSeq = []
        for rowId, row in enumerate(srcBatch):
            # get particular timestamp of all the batches together
            srcEmbedIndex = self.getIndex(row, inference=inference)
            # srcEmbedIndex is of dimension: batchsize (particular timestep word in all sentences of that batch)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out, self.enc_hidden = self.encoder(srcBatch.shape[1], srcEmbedIndex, self.enc_hidden)

            encoderOuts.append(enc_out.squeeze(0))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out, self.rev_hidden = self.revcoder(srcBatch.shape[1], srcEmbedIndex, self.rev_hidden)
                revcoderOuts.append(rev_out.squeeze(0))
            revcoderOuts.reverse()

        if self.cnfg.use_reverse:
            encoderOuts = [torch.add(x, y) for x, y in zip(encoderOuts, revcoderOuts)]

        if self.cnfg.srcMasking:  # True
            srcMaskTensor = torch.Tensor(srcMask)
            if torch.cuda.is_available():
                srcMaskTensor = srcMaskTensor.cuda()

            # isn't length of encoderOuts same as length of maskTensor always ?
            srcMaskTensor = torch.chunk(autograd.Variable(srcMaskTensor), len(encoderOuts), 0)
            srcMaskTensor = [x.contiguous().view(-1, 1) for x in srcMaskTensor]
            encoderOuts = [encoderOut * (x.expand(encoderOut.size())) for encoderOut, x in
                           zip(encoderOuts, srcMaskTensor)]
            del srcMaskTensor

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del enc_out

        zeroInit = torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit = zeroInit.cuda()
        c_0 = autograd.Variable(zeroInit)

        batch = batch.T
        # timestamps x batchsize_target
        batch_size = batch.shape[1]

        self.hidden = self.enc_hidden  # last hidden layer of encoder is the first hidden of decoder

        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed == False:  # dont use both forward and reverse
                if self.cnfg.init_enc:
                    self.hidden = self.enc_hidden
                else:
                    self.hidden = self.rev_hidden
            else:  # use both forward and reverse encoders to init decoder hidden
                if self.cnfg.use_LSTM:  # enc_hidden has 2 units
                    self.hidden = (torch.add(self.enc_hidden[0], self.rev_hidden[0]),
                                   torch.add(self.enc_hidden[1], self.rev_hidden[1]))
                else:
                    self.hidden = torch.add(self.enc_hidden, self.rev_hidden)

        encoder_seqlen = len(srcBatch)
        batch_size_src = len(srcBatch[0])

        zeroInit_1 = torch.zeros((batch_size_src, encoder_seqlen))
        if torch.cuda.is_available():
            zeroInit_1 = zeroInit_1.cuda()
        a_0 = autograd.Variable(zeroInit_1)

        # Init with START token
        if self.cnfg.use_attention:
            contextVectors = []
            attnweights = []
            contextVectors.append(c_0)
            if self.cnfg.pointer:
                attnweights.append(a_0)

        row = np.array([self.cnfg.start, ] * batch.shape[1])

        tgtEmbedIndex = self.getIndex(row, inference=inference)
        # forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
        out, self.hidden, c_0 = self.decoder(batch.shape[1], tgtEmbedIndex, None, None, self.hidden,
                                             feedContextVector=True, contextVector=c_0)

        if self.cnfg.mem_optimize:
            if not self.cnfg.context_dropout:
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        decoderOuts = [out.squeeze(0), ]
        tgts = []
        encoderOutTensor = torch.stack([encoderOut for encoderOut in encoderOuts], dim=0)
        for rowId, row in enumerate(batch):
            # iterate over timestamps of target side
            tgtEmbedIndex = self.getIndex(row, inference=inference)
            o_t = decoderOuts[-1]

            # forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
            if not self.cnfg.pointer:
                out, self.hidden, c_t = self.decoder(batch.shape[1], tgtEmbedIndex, encoderOutTensor, o_t, self.hidden,
                                                     feedContextVector=False)
            else:
                out, self.hidden, c_t, a_t = self.decoder(batch.shape[1], tgtEmbedIndex, encoderOutTensor, o_t,
                                                          self.hidden,
                                                          feedContextVector=False)
            # hidden layer passed as argument in next iteration

            tgts.append(self.getIndex(row))
            decoderOuts.append(out.squeeze(0))

            if self.cnfg.use_attention:
                contextVectors.append(c_t)
                if self.cnfg.pointer:
                    attnweights.append(a_t)

            if self.cnfg.mem_optimize:
                if self.cnfg.use_attention:
                    del c_t
                    if self.cnfg.pointer:
                        del a_t

        if self.cnfg.mem_optimize:
            del encoderOutTensor

        decoderOuts = decoderOuts[:-1]
        if self.cnfg.use_attention:
            contextVectors = contextVectors[:-1]

        if self.cnfg.use_attention and self.cnfg.use_downstream:  # Both True
            decoderOuts = [torch.cat([decoderOut, c_t], 1) for decoderOut, c_t in zip(decoderOuts, contextVectors)]

        if self.cnfg.mem_optimize:
            del encoderOuts
            del out
            del self.hidden
            if self.cnfg.use_attention:
                del contextVectors
            gc.collect()

        if not self.cnfg.useGumbel:  # False
            # totalLoss = sum(
            #     [loss_function(F.log_softmax(self.W(decoderOut)), tgt) for decoderOut, tgt in zip(decoderOuts, tgts)])

            if not self.cnfg.pointer:
                totalLoss = sum([loss_function(F.log_softmax(self.W(decoderOut)), tgt) for decoderOut, tgt in
                                 zip(decoderOuts, tgts)])
            else:
                loss = []
                srcBatch_tensor = torch.from_numpy(srcBatch)
                if torch.cuda.is_available():
                    srcBatch_tensor = srcBatch_tensor.cuda()
                # dim: seqlen x batch_size

                ts = 0
                # decoderOuts: timestamps x batch_size
                for decoderOut, tgt, attnwt in zip(decoderOuts, tgts, attnweights):
                    # iterate over time stamps
                    # print('decoderOut', len(decoderOut)): # batch size

                    if ts == 0:  # don't use a_0 which is a part of attnweights
                        l = loss_function(F.log_softmax(self.W(decoderOut)), tgt)
                        loss.append(l)
                        ts += 1
                        continue

                    logits = self.W(decoderOut)
                    # Todo: how to separately handle first attention a_0 in training and testing?
                    # Todo: fix ext_vocab_size: seq2seq summarizer: utils.py L206
                    output = torch.zeros(batch_size, self.cnfg.tgtVocabSize)
                    if torch.cuda.is_available():
                        output = output.cuda()

                    # distribute probabilities between generator and pointer
                    prob_ptr_logits = self.ptr(decoderOut)
                    prob_ptr = F.sigmoid(prob_ptr_logits)  # (batch size, 1)
                    prob_gen = 1 - prob_ptr
                    # add generator probabilities to output
                    gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
                    output[:, :self.cnfg.tgtVocabSize] = prob_gen * gen_output
                    # using source side vocab to generate words
                    # add pointer probabilities to output
                    ptr_output = attnwt
                    # batchsize x seq_len
                    output.scatter_add_(1, srcBatch_tensor.transpose(0, 1), prob_ptr * ptr_output)
                    output = torch.log(output + 1e-31)
                    l = loss_function(output, tgt)
                    loss.append(l)

                totalLoss = sum(loss)
        else:
            totalLoss = sum(
                [loss_function(self.gumbelMax(self.W(decoderOut)), tgt) for decoderOut, tgt in zip(decoderOuts, tgts)])

        return totalLoss

    def gumbelMax(self, scores, tau=1.1, temperature=0.1):
        noise = torch.rand(scores.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = autograd.Variable(noise)
        if torch.cuda.is_available():
            noise = noise.cuda()
        x = (scores + noise) / tau + temperature
        x = F.log_softmax(x.view(scores.size(0), -1))
        return x.view_as(scores)
