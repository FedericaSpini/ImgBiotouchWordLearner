class PyImgDataset:
    def __init__(self):
        self.writing_style_to_sessions = {}

    def add_session(self, writing_style, session):
        if not writing_style in self.writing_style_to_sessions:
            self.writing_style_to_sessions[writing_style] = [session]
        else:
            self.writing_style_to_sessions[writing_style] += [session]

    def getData(self):
        return self.writing_style_to_sessions

class PyImgSession:
    def __init__(self, user_id, session_number, images=[]):
        self.user_id = user_id
        self.images = images

    def add_image(self, image):
        self.images += [image]

    def getImages(self):
        return self.images
