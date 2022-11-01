FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install \
    apache2
COPY index.html /var/www/html/
RUN ls -R /var/www/html/
RUN cat /var/www/html/index.html
RUN echo "ServerName 34.90.222.126" >> /etc/apache2/apache2.conf
CMD /usr/sbin/apache2ctl -D FOREGROUND
EXPOSE 80

