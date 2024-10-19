import os
import sys
from main.utils.prepare import configs

if not os.path.exists(configs['upload_dir']):
    os.system(f"mkdir {configs['upload_dir']}")

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    send_from_directory,
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from main.classify import classify_image
from urllib.parse import quote
import secrets

app = Flask(__name__)
secret_key = secrets.token_hex(24)
with open('key.yml','w+') as key_w:
    key_w.write(f'Web_Secret_Key: "{secret_key}"\n')
app.secret_key = secret_key

@app.route("/", methods=["GET"])
def index():
    if os.listdir(uploaded_folder):
        for file in os.listdir(uploaded_folder):
            os.remove(os.path.join(uploaded_folder, file))
    return render_template("index.html")


# 使用绝对路径来确保路径的准确性
static = os.path.abspath(configs["static"])
uploaded_folder = os.path.abspath(configs["upload_dir"])


@app.route("/ur", methods=['POST'])
def upload_classify():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    pic = request.files["file"]

    if pic.filename == "":
        return jsonify({"error": "no selected file"}), 400

    if pic:
        try:
            # 保存上传的文件
            file_path = os.path.join(str(uploaded_folder), str(pic.filename))
            pic.save(file_path)

            result = classify_image(file_path)
            print(f'res: {result}')
            category = str(result[1][0])
            print(f'cate:{category}')
            encoded_category_name = quote(
                category.split(" ")[1] if len(category.split(" ")) > 1 else category
            )
            print(f'encoded_cname:{encoded_category_name}')

            session["result_image"] = pic.filename
            session["result_category"] = category
            session["result_encoded_category"] = encoded_category_name
            print('session End')
            return jsonify({"redirect": url_for("result")})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/result", methods=["GET"])
def result():
    image_filename = session.get("result_image")
    category = session.get("result_category")
    encoded_category = session.get("result_encoded_category")

    if not image_filename or not category:
        return redirect(url_for("index"))

    # 手动构造图片的 URL
    image_url = url_for("serve_uploaded", filename=image_filename)

    return render_template(
        "result.html",
        image=image_url,
        category=category,
        encoded_category_name=encoded_category,
    )


@app.route("/uploaded/<filename>")
def serve_uploaded(filename):
    return send_from_directory(uploaded_folder, filename)

@app.route("/static/<filename>")
def serve_static(filename):
    return send_from_directory(static, filename)

@app.route("/butterfly", methods=["GET"])
def butterfly():
    """
    显示蝴蝶介绍
    """
    url = "https://www.inaturalist.org/taxa/47224-Papilionoidea"
    return redirect(url)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8090, debug=True)
logging.basicConfig(level=logging.INFO)
