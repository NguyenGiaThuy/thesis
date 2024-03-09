# ============================================================================================
# Create OR Adopt the default VPC
# ============================================================================================
resource "aws_default_vpc" "default_vpc" {
  tags = {
    Name = "Default VPC"
  }
}

# ============================================================================================
# Security Group to allow port 22, 80, 443
# ============================================================================================
resource "aws_security_group" "tf_thesis_sg" {
  name        = "tf_thesis_sg"
  description = "Allow TLS inbound traffic and all outbound traffic"
  vpc_id      = aws_default_vpc.default_vpc.id

  tags = {
    Name = "tf_thesis_sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "tf_thesis_allow_ssh" {
  description       = "SSH"
  security_group_id = aws_security_group.tf_thesis_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 22
  ip_protocol       = "tcp"
  to_port           = 22

  tags = {
    Name = "tf_thesis_allow_ssh"
  }
}

resource "aws_vpc_security_group_ingress_rule" "tf_thesis_allow_http" {
  description       = "HTTP"
  security_group_id = aws_security_group.tf_thesis_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  ip_protocol       = "tcp"
  to_port           = 80

  tags = {
    Name = "tf_thesis_allow_http"
  }
}

resource "aws_vpc_security_group_ingress_rule" "tf_thesis_allow_https" {
  description       = "HTTPS"
  security_group_id = aws_security_group.tf_thesis_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  ip_protocol       = "tcp"
  to_port           = 443

  tags = {
    Name = "tf_thesis_allow_https"
  }
}

resource "aws_vpc_security_group_egress_rule" "tf_thesis_allow_all_outbound_ipv4" {
  security_group_id = aws_security_group.tf_thesis_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # semantically equivalent to all ports

  tags = {
    Name = "tf_thesis_allow_all_outbound_ipv4"
  }
}

resource "aws_vpc_security_group_egress_rule" "tf_thesis_allow_all_outbound_ipv6" {
  security_group_id = aws_security_group.tf_thesis_sg.id
  cidr_ipv6         = "::/0"
  ip_protocol       = "-1" # semantically equivalent to all ports

  tags = {
    Name = "tf_thesis_allow_all_outbound_ipv6"
  }
}

# ============================================================================================
# Instance
# ============================================================================================
resource "aws_instance" "tf_thesis_instance" {
  ami               = "ami-0123c9b6bfb7eb962"
  instance_type     = "t3.medium"
  availability_zone = "ap-southeast-1a"

  root_block_device {
    volume_size = 20
  }

  vpc_security_group_ids = [aws_security_group.tf_thesis_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              sudo apt update -y && sudo apt upgrade -y
              sudo apt install -y lsb-release gpg wget git vim zip
              
              # Install Docker, Docker Compose
              sudo apt update -y
              sudo apt install -y ca-certificates curl
              sudo install -m 0755 -d /etc/apt/keyrings
              sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
              sudo chmod a+r /etc/apt/keyrings/docker.asc
              echo \
                "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
                $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
                sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
              sudo apt update -y
              sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

              # Test Docker + Security Group + EIP
              sudo docker run -d --name instancetest -p 80:80 nginx

              EOF

  tags = {
    Name = "tf_thesis_instance"
  }
}

# ============================================================================================
# Elastic IP
# ============================================================================================
resource "aws_eip" "tf_thesis_eip" {
  domain = "vpc"
  instance = aws_instance.tf_thesis_instance.id

  tags = {
    Name = "tf_thesis_eip"
  }
}

# ============================================================================================
# Outputs
# ============================================================================================
output "tf_thesis_main_public_ip" {
  value = aws_eip.tf_thesis_eip.public_ip
}
