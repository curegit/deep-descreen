import typer

def main():

    	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit_code = 130
		return exit_code


def con():
    device = "cpu"


    #from .models import UNetLikeModel
    model = UNetLikeModel()
    model.load_state_dict(torch.load(sys.argv[2]))


    model.to(device)
    model.eval()
    print(model)

    patch_size = model.output_size(512)
    #img = read_uint16_image(sys.argv[3])

    with open(sys.argv[3], "rb") as fp:
        i = load_image(magickpng(fp.read(), png48=True), assert16=True)


    height, width = img.shape[1:3]
    # TODO: 4倍数にあわせる
    ppp_h = h % 512
    ppp_w = w % 512
    a_h = h + ppp_h
    a_w = w + ppp_w
    img = img.reshape((1, 3, h, w))
    res = np.zeros((3, a_h, a_w), dtype="float32")
    p = model.required_padding(patch_size)

    img = np.pad(img, ((0, 0), (0, 0), (p, p + ppp_h), (p, p + ppp_w)), mode="symmetric")
    for (j, i), (k, l) in model.patch_slices(a_h, a_w, patch_size):
        print(k)
        x = img[:, :, j, i]
        t = torch.from_numpy(x.astype("float32"))
        t = t.to(device)
        y = model(t)
        yy = y.detach().cpu().numpy()
        print(y.shape)
        res[:, k, l] = yy[0]
    res = res[:, :h, :w]
    save_image(res, sys.argv[4])
    #save_wide_gamut_uint16_array_as_srgb(res, sys.argv[4])
