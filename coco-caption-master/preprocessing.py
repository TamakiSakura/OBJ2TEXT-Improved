import json
resFile = './results/ptr_output.json'
res = json.load(open(resFile))
new_res = []
ids = []
j=0
for i in res:
	id = i["image_id"]
	j+=1
	if id not in ids:
		new_res.append(i)
		ids.append(id)
	if j%1000==0:
		print("Data %d"%(j))
with open('./results/ptr_output_new.json', 'w') as outfile:
	json.dump(new_res, outfile)