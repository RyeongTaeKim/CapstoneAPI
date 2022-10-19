from bs4 import BeautifulSoup
import requests
import os

# test 웹툰 url : https://comic.naver.com/webtoon/list?titleId=597447&weekday=sat
# test 웹툰 url : 제457화 킹 안드레 (2)

def crawl_cartoon(url, series_title):
    # Webtoon URL 
    url = url

    # 크롤링 우회
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'}
    html = requests.get(url, headers = headers)
    result = BeautifulSoup(html.content, "html.parser")

    # webtoonName = ['웹툰명', '\t\t\t\t\t\t작가명']
    webtoonName = result.find("span", {"class", "title"}).parent.get_text().strip().split('\n')

    # 현재 디렉토리의 하위 폴더에 이미지를 저장할 폴더 생성
    cwd = os.getcwd()
    files = os.listdir(cwd)

    # 현재 디렉토리 위치
    print(cwd, end='\n\n')

    #크롤링한 이미지를 저장할 폴더를 만듦
    if os.path.isdir(os.path.join(cwd, webtoonName[0])) == False:
        os.mkdir(webtoonName[0])

    print(webtoonName[0] + " folder created successfully!")
    os.chdir(os.path.join(cwd, webtoonName[0]))

    # title 클래스의 td 태그를 찾은 후, 최근 10회차에 대한 웹툰 이미지를 크롤링한다.
    title = result.find_all("td", {"class", "title"})

    for t in title:
        #크롤링하려는 회차의 이름과 동일한 경우에만 크롤링
        if (t.text).strip().split(" :")[0] == series_title:
            # 회차별로 디렉토리를 만든 후 해당 디렉토리로 이동
            os.mkdir((t.text).strip().split(" :")[0])
            os.chdir(os.getcwd() + "\\" + (t.text).strip().split(" :")[0])

            # 각 회차별 html 소스 가져오기
            url2 = "https://comic.naver.com" + t.a['href']
            html2 = requests.get(url2, headers = headers)
            result2 = BeautifulSoup(html2.content, "html.parser")

            # webtoon image 가져오기
            webtoonImg = result2.find("div", {"class", "wt_viewer"}).find_all("img")
            num = 1 # image_name

            for i in webtoonImg:
                saveName = os.getcwd() + "\\" + str(num) + ".jpg"
                with open(saveName, "wb") as file:
                    src = requests.get(i['src'], headers = headers)
                    file.write(src.content)
                num += 1
        
            os.chdir("..")

            # 한 회차 이미지 저장 완료!
            print((t.text).strip() + "  saved completely!")
    
    os.chdir("..")