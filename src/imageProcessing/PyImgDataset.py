class PyImgDataset:
    def __init__(self):
        self.writing_style_to_sessions = {}
        # self.classes_to_samples_dicts = {} #This dictionary maps each Handwriting_style to a dict, wich maps class to sample list
                                            # So { "BLOCK_LETTERS :{"u1":[u1 sample list], ..., "uN":[uN sample list]} , "ITALIC":{...}}

    def add_session(self, writing_style, session):
        if not writing_style in self.writing_style_to_sessions:
            self.writing_style_to_sessions[writing_style] = [session]
            # self.classes_to_samples_dicts[writing_style] =
        else:
            self.writing_style_to_sessions[writing_style] += [session]

    def getData(self):
        return self.writing_style_to_sessions

    # def get_class_to_samples_dict(self, writing_style):
    #     for session in self.writing_style_to_sessions[writing_style]:
    #         print("Ciao")
            # if session.user_id not in self.classes_to_samples_dict:
            #     self.classes_to_samples_dict[session.user_id] = session.get_only_imgs()
            #     # print("\n\n\n\n")
            #     # print(type(session.images[0]))
            # else:
            #     self.classes_to_samples_dict[session.user_id] += [session.get_only_imgs()]
        # print(self.classes_to_samples_dict)

