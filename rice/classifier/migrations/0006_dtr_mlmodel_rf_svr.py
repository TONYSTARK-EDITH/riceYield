# Generated by Django 3.2 on 2021-05-22 11:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('classifier', '0005_auto_20210522_1727'),
    ]

    operations = [
        migrations.CreateModel(
            name='DTR',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'DTR',
            },
        ),
        migrations.CreateModel(
            name='Mlmodel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nitrogen', models.FloatField()),
                ('phosphorus', models.FloatField()),
                ('pottasium', models.FloatField()),
                ('rainfall', models.FloatField()),
                ('rice_yield', models.FloatField(null=True)),
            ],
            options={
                'verbose_name_plural': 'MlModel',
            },
        ),
        migrations.CreateModel(
            name='RF',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'RF',
            },
        ),
        migrations.CreateModel(
            name='svr',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'SVR',
            },
        ),
    ]
