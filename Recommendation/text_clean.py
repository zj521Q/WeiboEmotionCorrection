import re
import jieba as jb

_stopwords = None

def load_stopwords(file_path='hit_stopwords.txt'):
    global _stopwords
    if _stopwords is None:
        with open(file_path, 'r', encoding='utf-8') as f:
            _stopwords = [line.strip() for line in f if line.strip()]
    return _stopwords

def clean_text(text):
    pattern = (
        r'http[s]?://\S+|'              
        r'回复@[a-zA-Z\u4e00-\u9fa5_0-9-]+|'  
        r'@[a-zA-Z\u4e00-\u9fa5_0-9-]+|'      
        r'抱歉，此微博已被作者删除|'              
        r'\d{4}年\d{1,2}月\d{1,2}日|'          
        r'\d{4}年\d{1,2}月|'                 
        r'\d{1,2}月\d{1,2}日|'                
        r'转发微博|'                          
        r'转发理由|'                        
        r'转发内容|'                          
        r'原始用户:\s*[^\n]+|'               
        r'原图|'                             
        r'\[组图共\d+张\]'                   
        r'(?:\S+的)?微博(?:视频|图片)|'         
        r'转发|'                             
        r'微博正文|'
        r'[\w\u4e00-\u9fff]+?的微博视频|'
        r'超话|'
        r'展开|'
        r'分享图片|'
        r'发布头条文章|'
        r'抱歉，?由于作者设置，?你暂时没有这条微博的查看权限哦。?查看帮助：?.*?(?=\s|$)'
    )
    text = re.sub(pattern, '', text)
    rule = re.compile(u"[^ \u4E00-\u9FA5]")
    text = rule.sub('', text.replace('\n', ''))
    if not re.search(r'[\u4e00-\u9fa5]', text):
        return ''
    tokens = jb.lcut(text)
    stopwords = load_stopwords()
    cleaned_tokens = [token.strip() for token in tokens if token.strip() and token not in stopwords]
    return ' '.join(cleaned_tokens)