from Rocket import Rocket
import rkt_config

if __name__ == "__main__":
    starship = Rocket(rkt_config.config)
    starship.run(mode='MIL')