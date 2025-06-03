from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,

    FlexMessage,
    FlexContainer
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    FollowEvent,
    PostbackEvent
)

import os
import datetime 
import json
import numpy as np
import requests
from PIL import Image

app = Flask(__name__)
# V2
# configuration = Configuration(access_token = 'GSRKbLH7evIuhdoUwX+zvpar8/aWzCzfvdivP9Pi2Hg0m+Ivn0P0wadcyzhRcANCQPEbitC7Ncke1DyiMuWRcT3rwZ3UH4stm/q0NZ+sjnrM+b0mKWKDL/lP6Gpygmf45qILvmYgYUL6u6Rik0GBgAdB04t89/1O/w1cDnyilFU=')
# handler = WebhookHandler('e485d86db895039e826724b9e0eeb8d4')

# V3
configuration = Configuration(access_token = 'YNbN/yBpYf32HIwGPEml9opwW2uLpyYhSP4psKzXdvltubXdiVZ5XHeTF1HB0/nrzpEQYE3SydWJJ0ewsC23SGaKh0qyDVksxZNjkrRVT6p0NDlAjoUqTLEhZlJfesw41k9z9I7n5JK1RZev62rQ7gdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('93ead61523bb5b6325a919ef6e6fbcfa')

# 大喜利を読み込み
with open("/home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/Generated_Bokes/all_bokes.json", "r") as f:
    BOKES = json.load(f)
IMAGE_DIR = "/home/user/workspace/GUMI_T_Paper/datas/Bokete_Dataset/boke_image/"

# 動作確認
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    print("verified")
    return 'OK'

## 友達追加された場合
@handler.add(FollowEvent)
def handle_follow(event):

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

    # 年齢・性別を投票するボタン
    bubble_string = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
            {
                "type": "text",
                "text": "あなたの属性について答えてください．",
                "weight": "bold",
                "size": "md"
            },
            {
                "type": "text",
                "text": "間違えた項目を選択した場合，"
            },
            {
                "type": "text",
                "text": "改めてタップしてください．"
            }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                # ボタン
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "10代・男性",
                        "data": "profile:10,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "10代・女性",
                        "data": "profile:10,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "20代・男性",
                        "data": "profile:20,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "20代・女性",
                        "data": "profile:20,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "30代・男性",
                        "data": "profile:30,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "30代・女性",
                        "data": "profile:30,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "40代・男性",
                        "data": "profile:40,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "40代・女性",
                        "data": "profile:40,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "50代・男性",
                        "data": "profile:50,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "50代・女性",
                        "data": "profile:50,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "10代・男性",
                        "data": "profile:10,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "60代・女性",
                        "data": "profile:60,female"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "70代・男性",
                        "data": "profile:70,male"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "70代・女性",
                        "data": "profile:70,female"
                    }
                },                    
            ],
            "flex": 0
        }
    }
    message = FlexMessage(alt_text = "hello", contents = FlexContainer.from_dict(bubble_string))

    text = """人工知能が画像に対して大喜利文章を生成します．

送信される大喜利に対して面白さを評価してください！

☆☆☆☆☆使い方☆☆☆☆☆
1.　年齢・性別をタップしてください．
2.　大喜利の面白さをタップしてください．面白さを評価すると次の大喜利が送信されます！
※　いずれも，誤った項目をタップした場合，正しい項目をタップし直してください．
☆☆☆☆☆☆☆☆☆☆☆☆☆

ボケて（bokete），株式会社オモロキとは一切関わりのないものが運営しています．
連絡先 : 25T2501@sit.shonan-it.ac.jp
"""

    line_bot_api.reply_message(ReplyMessageRequest(
        replyToken = event.reply_token,
        messages = [ message, TextMessage(text = text) ]
    ))

def choice_ohgiri():
    
    with open("/home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/current_index.txt", "r") as f:
        current_index = int(f.read().strip())

    current_index = current_index % len(BOKES)
    ID = BOKES[current_index][0]
    image_number = int(ID.split("_")[1])
    method = ID.split("_")[0]
    sentence = BOKES[current_index][1]
    star = BOKES[current_index][2]
    
    
    # 画像をアップロード
    image_path = f"{IMAGE_DIR}{image_number}.jpg"
    image = Image.open(image_path).resize((512, 512))
    image.save("tmp.jpg")
    print(image.size)
    with open("tmp.jpg", "rb") as f:
        files = {'imagedata':f.read()}

    print(method, sentence, image_number, star)

    headers = {'Authorization': "Bearer {}".format("Br5uXr6Mtu3cxDFGF9nZO6ahUEUHZJQ9mb1bzUTydW4")}
    URL = "https://upload.gyazo.com/api/upload"
    response = requests.request('post', URL, headers = headers, files = files)
    image_link = response.json()["url"]

    button = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": image_link,
            "size": "full",
            "aspectRatio": "20:20",
            "aspectMode": "cover",
            "action": {
            "type": "uri",
            "uri": "https://line.me/"
            }
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
            {
                "type": "text",
                "text": sentence,
                "weight": "bold",
                "size": "xl",
                "wrap": True
            }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "面白い",
                        "data": f"vote:3,{method},{image_number},{sentence},{star}"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "やや面白い",
                        "data": f"vote:2,{method},{image_number},{sentence},{star}"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "ややつまらない",
                        "data": f"vote:1,{method},{image_number},{sentence},{star}"
                    }
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "つまらない",
                        "data": f"vote:0,{method},{image_number},{sentence},{star}"
                    }
                },
            ],
            "flex": 0
        }
    }

    with open("/home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/current_index.txt", "w") as f:
        f.write(str(current_index + 1))

    return button

# メッセージが送られた場合
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:

        button = choice_ohgiri()
        message = FlexMessage(alt_text="hello", contents=FlexContainer.from_dict(button))

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages = [
                        # TextMessage(text=event.message.text), 
                        message
                    ]
            )
        )

# 投票された場合
@handler.add(PostbackEvent)
def handle_postback(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        response = event.postback.data

        # 年齢・性別が投票された場合
        if "profile:" in response:
            user_id = event.source.user_id

            # 年齢・性別をCSVに記述
            with open("user_informations.csv", "a") as f:
                age_gender = response.split("profile:")[-1]
                f.write(f"{age_gender},{user_id}\n")

        # ボケに対して投票された場合
        elif "vote:" in response:

            user_id = event.source.user_id

            now = datetime.datetime.now()
            formatted_now = now.strftime("%Y/%m/%d,%H:%M:%S")

            # 投票結果をCSVに記述
            with open("vote_results.csv", "a") as f:
                rate_model_name_image_number = response.split("vote:")[-1]
                f.write(f"{formatted_now},{rate_model_name_image_number},{user_id}\n")
        
        button = choice_ohgiri()
        message = FlexMessage(alt_text = "hello", contents = FlexContainer.from_dict(button))

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages = [
                        # TextMessage(text = response), 
                        message
                    ]
            )
        )

if __name__ == "__main__":
    app.run()

"""
1つ目のコマンドプロンプトで
    wsl -d Ubuntu2204_Colab20250113 -u user
    cd /home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs
    source /home/user/Colab_20250113/bin/activate
    flask run --reload --port 8080 --without-threads
2つ目のコマンドプロンプトで
    wsl -d Ubuntu2204_Colab20250113 -u user
    ngrok http 8080
を実行

PermissionErrorが出る場合は，
    sudo chmod 777 /home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/vote_results.csv
    sudo chmod 777 /home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/user_informations.csv
    sudo chmod 777 /home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/tmp.jpg
    sudo chmod 777 /home/user/workspace/GUMI_T_Paper/GUMI_T_Paper_programs/current_index.txt
を実行

    https://
のＵＲＬを，
    https://developers.line.biz/console/channel/
のＷｅｂｈｏｃｋに指定（末尾に/callbackを付ける）

https://qiita.com/mintak21/items/fe9234d4b6a98bfc841a
https://dashboard.ngrok.com/get-started/setup/linux
"""