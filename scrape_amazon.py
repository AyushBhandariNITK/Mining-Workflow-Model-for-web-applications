
import requests
from parsel import Selector
from numpy import loadtxt

import time
start = time.time()



response = requests.get('https://www.amazon.in/')

## Setup for scrapping tool

# "response.txt" contain all web page content
selector = Selector(response.text)

# Extracting href attribute from anchor tag <a href="*">
href_links = selector.xpath('//a/@href \t').getall()



#Extracting src attribute from img tag <img src="*">
image_links = selector.xpath('//img/@src').getall()


filename = '/home/spokharna/5thsem/se/project/TwitterAnalytics/code/config/Links.txt'

with open(filename, 'w') as file_object:
    file_object.write(str("\n\nHref Links\n"))
    file_object.write(str(href_links))
print('\nhref_links of amazon are stored in file "Links.txt"\n')

with open(filename, 'a') as file_object:
    file_object.write(str("\n\nImage links\n"))
    file_object.write(str(image_links))
# print(image_links)
print('\nimage_links of amazon are stored in file "Links.txt"\n')


end = time.time()
print("Time taken in seconds : ", (end-start))

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from multiprocessing import Process, Queue, Pool, Manager
import threading
import sys
import plotly.express as px


proxies = {
  'http': 'http://134.119.205.253:8080',
  'https': 'http://134.119.205.253:8080',
}

startTime = time.time()
qcount = 0
products=[] #List to store name of the product
prices=[] #List to store price of the product
ratings=[] #List to store ratings of the product
no_pages = 20


def get_data(pageNo,q):  
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    
    #different links on which we tried but screenshot are of laptop page.
    r = requests.get("https://www.amazon.com/s?k=laptops&page="+str(pageNo), headers=headers)#, proxies=proxies)
    # s = requests.get("https://www.amazon.com/s?k=earphones&page="+str(pageNo), headers=headers)#, proxies=proxies)
    content = r.content
    # content = s.content
    soup = BeautifulSoup(content,"lxml")
    
    
    for d in soup.findAll('div', attrs={'class':'sg-col-4-of-12 sg-col-8-of-16 sg-col-16-of-24 sg-col-12-of-20 sg-col-24-of-32 sg-col sg-col-28-of-36 sg-col-20-of-28'}):
        name = d.find('span', attrs={'class':'a-size-medium a-color-base a-text-normal'})
        price = d.find('span', attrs={'class':'a-offscreen'})
        rating = d.find('span', attrs={'class':'a-icon-alt'})
        all=[]
        
        if name is not None:
            all.append(name.text)
        else:
            all.append("unknown-product")
 
        if price is not None:
            all.append(price.text)
        else:
            all.append('$0')
        
        if rating is not None:
            all.append(rating.text)
        else:
            all.append('-1')
        q.put(all)
        #print("---------------------------------------------------------------") 
results = []
if __name__ == "__main__":
    m = Manager()
    q = m.Queue() # use this manager Queue instead of multiprocessing Queue as that causes error
    p = {}
    if sys.argv[1] in ['t', 'p']: # user decides which method to invoke: thread, process or pool
        for i in range(1,no_pages):
            if sys.argv[1] in ['t']:
                print("starting thread: ",i)
                p[i] = threading.Thread(target=get_data, args=(i,q))
                p[i].start()
            elif sys.argv[1] in ['p']:
                print("starting process: ",i)
                p[i] = Process(target=get_data, args=(i,q))
                p[i].start()

        # join should be done in seperate for loop 
        # reason being that once we join within previous for loop, join for p1 will start working
        # and hence will not allow the code to run after one iteration till that join is complete, ie.
        # the thread which is started as p1 is completed, so it essentially becomes a serial work instead of 
        # parallel

        for i in range(1,no_pages):
            p[i].join()
    else:
        pool_tuple = [(x,q) for x in range(1,no_pages)]
        with Pool(processes=8) as pool:
            print("in pool")
            results = pool.starmap(get_data, pool_tuple)
    
    while q.empty() is not True:
        qcount = qcount+1
        queue_top = q.get()
        products.append(queue_top[0])
        prices.append(queue_top[1])
        ratings.append(queue_top[2])
        
    print("total time taken: ", str(time.time()-startTime), " qcount: ", qcount)
    #print(q.get())
    df = pd.DataFrame({'Product Name':products, 'Price':prices, 'Ratings':ratings})
    print(df)
    df.to_csv('products.csv', index=False, encoding='utf-8')
    df.sort_values(["Price","Product Name"],axis=0,ascending=True,inplace=True)
    fig=px.line(df,x="Product Name",y="Price",title="Graph")
    fig.show()
    fig=px.line(df,x="Product Name",y="Ratings",title="Graph")
    fig.show()
