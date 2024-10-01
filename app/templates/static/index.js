document.addEventListener("DOMContentLoaded", function () {
  console.log("DOM fully loaded and parsed");

  // 点击 "Upload Picture" 按钮时，触发隐藏的文件输入
  document
    .getElementById("uploadButton")
    .addEventListener("click", function () {
      document.getElementById("imageInput").click(); // Trigger file input click
    });

  // 当选择文件时显示预览
  document
    .getElementById("imageInput")
    .addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          const preview = document.getElementById("imagePreview");
          preview.src = e.target.result;
          preview.style.display = "block"; // Show image preview
        };
        reader.readAsDataURL(file);
      }
    });

  // 上传文件
  window.uploadFile = function () {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];
    const formData = new FormData();

    if (file) {
      formData.append("file", file);

      fetch("/ur", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.redirect) {
            window.location.href = data.redirect; // Redirect to the result page
          } else {
            const photoResult = document.getElementById("photoResult");
            photoResult.innerHTML = `<p>Error: ${data.error}</p>`;
          }
        })
        .catch((error) => {
          const photoResult = document.getElementById("photoResult");
          photoResult.innerHTML = `<p>Error: ${error}</p>`;
        });
    } else {
      alert("Please select a file.");
    }
  };

  // 点击 "Take Photo" 按钮时，触发隐藏的文件输入（相机）
  window.takePhoto = function () {
    const cameraInput = document.getElementById("cameraInput");
    cameraInput.click(); // Trigger the file input to open the camera

    cameraInput.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        fetch("/ur", {
          method: "POST",
          body: formData,
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.redirect) {
              window.location.href = data.redirect; // Redirect to result page
            } else if (data.error) {
              alert("Error: " + data.error);
            }
          })
          .catch((error) => {
            alert("Error: " + error);
          });
      } else {
        alert("Please select a file.");
      }
    });
  };

  // 当选择文件时显示预览
  document
    .getElementById("cameraInput")
    .addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          const preview = document.getElementById("photoPreview");
          preview.src = e.target.result;
          preview.style.display = "block"; // Show the photo preview
        };
        reader.readAsDataURL(file);
      }
    });
});
