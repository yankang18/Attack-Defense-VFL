

def get_passport_config():
    # ====== "VHNN" ======
    # passport_args = {"top_model_force_passport": {'0': False, '1': False},
    #                  "active_model_force_passport": {'0': False, '1': False, '2': False},
    #                  "passive_model_force_passport": passive_pp_args}
    passport_args = {"top_model_force_passport": {0: False, 1: False},
                     "active_model_force_passport": {0: False, 1: False},
                     "passive_model_force_passport": {0: False, 1: False}}

    # ====== "VSNN" ======
    # passport_args = {"top_model_force_passport": {'0': False, '1': False},
    #                  "active_model_force_passport": None,
    #                  "passive_model_force_passport": {'0': False, '1': False, '2': False}}
    # passport_args = {"top_model_force_passport": {'0': False, '1': False},
    #                  "active_model_force_passport": None,
    #                  "passive_model_force_passport": {'0': False, '1': False}}
    return passport_args