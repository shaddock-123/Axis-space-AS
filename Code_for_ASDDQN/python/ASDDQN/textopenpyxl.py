import xlwt

if __name__ == "__main__":
    #创建样式
    style = xlwt.XFStyle()  # 初始化样式
    font = xlwt.Font() # 为样式创建字体
    font.name = 'Time New Roman'
    font.bold = True
    font.height = 15*10
    style.font = font
    #写入excel
    step =0
    reward = 3
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet(sheetname='sheet1')
    text = ["step","reward"]
    for i in range(1,2):
       sheet1.write(0,i,text[i-1],style)
    for j in range(1,100):
        step += 1
        reward  += 2
        sheet1.write(j,0,step,style)
        sheet1.write(j,1,reward,style)
    workbook.save('emo.xls')
