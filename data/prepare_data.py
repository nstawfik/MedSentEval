import sys, urllib, re, json, socket, string
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

#preprocess RQE
tree = ET.parse("RQE/RQE_Train_8588_AMIA2016.xml")
root = tree.getroot()
x=[]
chqs=[]
faqs=[]
file=open('RQE/rqe_train.txt', 'w')
for member in root.findall('pair'):
  pid=member.attrib['pid']
  val= member.attrib['value']
  c = member.find('chq').text.rstrip().lstrip()
  f = member.find('faq').text.rstrip().lstrip()
  file.write(pid+"\t"+val+"\t"+c+"\t"+f+"\n")
file.close()
tree = ET.parse("RQE/RQE_Test_302_pairs_AMIA2016.xml")
root = tree.getroot()
x=[]
chqs=[]
faqs=[]
file=open('RQE/rqe_test.txt', 'w')
for member in root.findall('pair'):
  pid=member.attrib['pid']
  val= member.attrib['value']
  c = member.find('chq').text.rstrip().lstrip()
  f = member.find('faq').text.rstrip().lstrip()
  file.write(pid+"\t"+val+"\t"+c+"\t"+f+"\n")
file.close()



#pre-process BIOC
root = ET.parse(r'BIOC/corpus.xml').getroot()
file = open("BIOC/train.txt","w") 
quest=""
for review in root:
    for pmid in review:
        if quest!=pmid.attrib['QUESTION']:
            quest=pmid.attrib['QUESTION']
        file.write(quest+"\t"+pmid.text+"\t"+pmid.attrib['ASSERTION']+"\n")
        
#pre-process VaccineSA 
file = open("ClinicalSA/train_temp.txt","r") 
l=file.readlines()
file.close()
id1,text,id2,label=[],[],[],[]
for line in l:
  id,text1=line.split("\t")
  id1.append(id)
  text.append(text1.rstrip())
file = open("ClinicalSA/AnnotationResults/TweetsAnnotation.txt","r") 
t=file.readlines()
for i in t:
  id,xx,label1=i.split("\t")
  id2.append(id)
  label.append(label1.rstrip())
file.close()
all=[]
for i in range(len(id1)):
     if (id1[i]==id2[i]):
         s=id1[1]+"\t"+label[i]+"\t"+text[i]
         all.append(s) 

file = open("ClinicalSA/train.txt","w") 
for line in all:
  i,ll,txt=line.split("\t")
  if len(txt)>1:
    file.write(line+"\n")
  
file.close()

