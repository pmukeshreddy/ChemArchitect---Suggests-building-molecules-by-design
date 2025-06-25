class PropertyEncoder:
    def __init__(self):
        self.properties = [
            'bioactive', 'pharmaceutical', 'insecticide', 'antibacterial',
            'antihypertensive', 'receptor_agonist', 'pharmaceutical_research'
        ]
        self.properties_idx = {prop:idx for prop,idx in self.properties.items()}
        self.num_properties = len(self.properties_idx)
    def encode_application(self,application_list):
        if isinstance(application_list,str):
            application_list = [application_list]
            
        vectors = torch.zeros(self.num_properties)

        for app in application_list:
            app_lower = app.lower()
            for prop in self.properties:
                if prop in app_lower:
                    idx = self.properties_idx[prop]
                    vectors[idx] = 1.0
        return vectors
    def encode_text(self,text):
        return self.encode_application(text)
