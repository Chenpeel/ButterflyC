import tensorflow as tf

def parse_function(filename, label, image_size):
    if image_size is None:
        raise ValueError("image_size cannot be None")
    
    filename = tf.strings.as_string(filename)
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image

def parse_and_augment(filename, label, image_size, num_augments):
    image, label = parse_function(filename, label, image_size)
    augmented_images = [augment(image) for _ in range(num_augments)]
    augmented_images = tf.stack(augmented_images)
    labels = tf.fill([num_augments], label)
    return augmented_images, labels

def get_batch(images, labels, image_size, batch_size, capacity, num_augments=3):
    if image_size is None:
        raise ValueError("image_size cannot be None")
    
    def process_image_and_label(filename, label):
        images, labels = parse_and_augment(filename, label, image_size, num_augments)
        return tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.flat_map(process_image_and_label)
    dataset = dataset.shuffle(buffer_size=capacity)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

