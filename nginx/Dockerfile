FROM nginx:1.19.0-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d

COPY 33CB825D7B29ACAB4CEB064C22409988.txt /var/www/.well-known/pki-validation/