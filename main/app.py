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
import os
import logging
from main.utils.prepare import configs
from main.recognize import recognize
from urllib.parse import quote

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    if os.listdir(uploaded_folder):
        for file in os.listdir(uploaded_folder):
            os.remove(os.path.join(uploaded_folder, file))
    return render_template("index.html")


# 使用绝对路径来确保路径的准确性
static = os.path.abspath(configs["static"])
uploaded_folder = os.path.abspath(configs["uploaded_picture"])


@app.route("/ur", methods=["POST"])
def upload_recognize():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    pic = request.files["file"]

    if pic.filename == "":
        return jsonify({"error": "no selected file"}), 400

    if pic:
        try:
            # 保存上传的文件
            file_path = os.path.join(uploaded_folder, pic.filename)
            pic.save(file_path)

            result = recognize(file_path)
            category = str(result[1][0])
            encoded_category_name = quote(
                category.split(" ")[1] if len(category.split(" ")) > 1 else category
            )

            # 在 session 中存储结果
            session["result_image"] = pic.filename
            session["result_category"] = category
            session["result_encoded_category"] = encoded_category_name
            # 返回 JSON 响应，其中包含重定向的 URL
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
    app.run(host="0.0.0.0",port=5000, debug=False)

logging.basicConfig(level=logging.INFO)
