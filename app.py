import os
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

#Import existing ADAIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image', validators=[InputRequired()])
    style = FileField('Style Image', validators=[InputRequired()])
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha (0.0 - 1.0)', default=1.0)
    submit = SubmitField('Transfer Style')


if torch.backends.mps.is_available():
        device = torch.device("mps")
elif torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")
        
        
encoder = VGGEncoder(BASE_DIR/ 'model_weights'/'vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(BASE_DIR / 'model_weights' / 'decoder_epoch_135.pth', map_location=device))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image, style_image, alpha, encoder, decoder, device):
    content_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    style_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        content_features = encoder(content_image, is_test=True)
        style_features = encoder(style_image, is_test = True)
        
        stylized_features = adaptive_instance_normalization(content_features, style_features)
        stylized_features = alpha * stylized_features + (1 - alpha) * content_features
        
        stylized_image = decoder(stylized_features)
        
        return stylized_image
    
    
def save_image(tensor, path):
    image = tensor.cpu().clone().squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
    
    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            else:
                content_filename = form.content_path.data
            
        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            else:
                style_filename = form.style_path.data
                
        
        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                alpha = form.alpha.data
                stylized_image = style_transfer(content_image, style_image, alpha, encoder, decoder, device)
                
                result_filename = f'stylized_{content_filename}'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                
                save_image(stylized_image, result_path)
                result_image = result_filename
                
            except Exception as e:
                error = f"Error processing images: {str(e)}"
    elif request.method == 'POST':
        if not content_filename:
            error = "Please upload a content image."
        if not style_filename:
            error = "Please upload a style image."
            
    return render_template('index.html', form=form, result_image=result_image, content_filename=content_filename, style_filename=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)