def load_txt(txt_path,img_width,img_height):
    boxes=[]
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data=line.strip().split()
            class_id=int(data[0])
            id=int(data[1])
            cx=float(data[2])
            cy=float(data[3])
            cw=float(data[4])
            ch=float(data[5])
            ##返归一化
            x=cx*img_width
            y=cy*img_height
            w=cw*img_width
            h=ch*img_height
            boxes.append([class_id,id,x,y,w,h])
    return boxes