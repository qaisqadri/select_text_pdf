import fitz

def select_words(inpath='',outpath='',words=dict()):
### READ IN PDF
	doc = fitz.open(inpath)
	color={'yellow':(0.90, 0.90, 0.0),'red':(1.0, 0.0, 0.0)}
	
	N=len(words) # 2 here
	for i in range(N):
		t=words.popitem()
		key=t[0]
		val_list=t[1]
		if key == 'good':
			col='yellow'
		else:
			col='red'

		for text in val_list:

			for x in range(doc.pageCount):
				page = doc[x]
				text_instances = page.searchFor(text)

				### HIGHLIGHT
				for inst in text_instances:
					annot = page.addRectAnnot(inst)
					annot.setColors({"fill":color.get(col)})
					annot.setBorder({'width':0.01})
					annot.setOpacity(0.4)


	doc.save(outpath, garbage=4, deflate=True, clean=True)

if __name__ == '__main__':
	select_words(inpath='in.pdf',outpath='out.pdf',words={'good':['text','page','elements whose'],'bad':['which','from','used']})
	# select_words(inpath='in.pdf',outpath='out.pdf',words={'good':['text','page'],'bad':['which','from','used'],'neutral':['of','the']})