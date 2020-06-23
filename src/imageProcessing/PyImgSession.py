class PyImgSession:
    def __init__(self, user_id, s_n, style):
        self.user_id = user_id
        self.session_number = s_n
        self.images = []
        self.writing_style = style

        print("\n\n\n\n")
        print(self.user_id, self.session_number, self.writing_style)

    def add_image(self, image):
        self.images += [image]
        # print(len(self.images))

    def getImages(self):
        return self.images

    def get_only_imgs(self):
        res = []
        print(len(self.images))
        for img in self.images:
            # print("\n\n\n\n")
            # print(len(img[0]))
            res += [img[0]]
        print(len(res))