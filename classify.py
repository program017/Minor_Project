import tensorflow as tf
import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cwd=os.getcwd()
path=cwd+"/images/"
#image_path = sys.argv[1]
flist=os.listdir(path)
result={}
times=0
label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()	
        graph_def.ParseFromString(f.read())	
        _ = tf.import_graph_def(graph_def, name='')
for loc in flist:
    image_path=path+loc
    times+=1
    print(image_path,times)
    #break
    image_data = tf.gfile.FastGFile(image_path, 'rb').read() 
    

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	# return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, 

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
    
	
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]*100
        if human_string in result:
            result[human_string]+=score
        else:
            result[human_string]=score
    sess.close()
        #print('%s (score = %.5f)' % (human_string, score))
#print(result)
final={}
for i in result:
        final[i]=result[i]/times
        #print(result[i])
#print(final)
female={}
female['anne hathaway']=0
female['kate winslet']=0
female['scarlett johanasson']=0
female['Emma Stone']=0
female['Jennifer Lawrence']=0
male={}
male['benedict cumberbatch']=0
male['leonardo dicaprio']=0
male['matthew mcconaughey']=0
male['ryan gosling']=0
male['ryan renolds']=0
male['jake gyllenhaal']=0
male['robert downey jr']=0
for i in final:
        if i in male:
                male[i]=final[i]*100
        else:
                female[i]=final[i]*100
mx=0
ind=0
tot=0
for i in male:
        tot+=male[i]

for i in male:
        male[i]=float(male[i])/float(tot)*100
tot=0
for i in female:
        tot+=female[i]

for i in female:
        female[i]=float(female[i])/float(tot)*100
        
for i in male:
        if mx==0 and ind==0:
                mx=male[i]
                ind=i
        else:
                if male[i]>mx:
                        mx=male[i]
                        ind=i
mx=0
ind2=0
for i in female:
        if mx==0 and ind==0:
                mx=female[i]
                ind2=i
        else:
                if female[i]>mx:
                        mx=female[i]
                        ind2=i
print("Leading male cast: "+str(ind)+" : "+str(male[ind]))
print("Leading female cast: "+str(ind2)+" : "+str(female[ind2]))
